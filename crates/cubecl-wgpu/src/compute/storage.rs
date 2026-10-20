use crate::compute::device_poison::PoisonWatch;
use cubecl_core::server::{IoError, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_environment::future::channel::Receiver;
use cubecl_server::memory_management::{ManagedMemoryBinding, PageGuard};
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{
    num::NonZeroU64,
    ops::{Deref, Range},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// The mapping span that covers `offset..offset + size`, and where the logical
/// slice starts in it. WGPU requires the start to be `MAP_ALIGNMENT`-aligned and
/// the length to be a multiple of `COPY_BUFFER_ALIGNMENT`.
fn host_mapping_range(offset: u64, size: u64) -> (Range<u64>, usize) {
    let start = offset & !(wgpu::MAP_ALIGNMENT - 1);
    let end = (offset + size).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
    (start..end, (offset - start) as usize)
}

/// Why the host cannot take a device allocation, or the GPU cannot take it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostAccessError {
    /// The allocation was created without host-visible primary memory enabled.
    DeviceLocal,
    /// The allocation already has a host mapping alive.
    AlreadyMapped,
    /// The allocation is mapped by the host, so a GPU write naming it is refused.
    Leased,
    /// WGPU failed to complete a mapping.
    MapFailed(String),
}

impl core::fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DeviceLocal => {
                f.write_str("the WGPU allocation is device-local and cannot be mapped by the host")
            }
            Self::AlreadyMapped => {
                f.write_str("the WGPU allocation already has an active host mapping")
            }
            Self::Leased => f.write_str(
                "the WGPU allocation is mapped by the host, so the GPU write naming it was refused",
            ),
            Self::MapFailed(reason) => {
                write!(f, "the WGPU host mapping failed: {reason}")
            }
        }
    }
}

impl std::error::Error for HostAccessError {}

impl From<HostAccessError> for ServerError {
    fn from(error: HostAccessError) -> Self {
        Self::HostAccessRefused {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

/// The host lease on one physical wgpu buffer.
///
/// A zero-copy read holds the lease for as long as its
/// [`Bytes`](cubecl_common::bytes::Bytes) lives. A launch
/// or write that names the buffer while leased is refused rather than waiting:
/// waiting on the device thread would block every other stream behind a host
/// reference held by arbitrary caller code.
///
/// ponytail: the lease is per wgpu buffer, so two concurrent zero-copy reads of
/// disjoint slices that share a pooled page are refused. A consumer that reads
/// many tensors at once can use `MemoryConfiguration::ExclusivePages`, where
/// every allocation is its own buffer; sharing one mapping between views would
/// lift the ceiling.
#[derive(Debug, Default)]
pub(crate) struct HostAccess {
    leased: AtomicBool,
}

impl HostAccess {
    fn is_leased(&self) -> bool {
        self.leased.load(Ordering::Acquire)
    }

    fn try_lease(&self) -> Result<(), HostAccessError> {
        self.leased
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map(|_| ())
            .map_err(|_| HostAccessError::AlreadyMapped)
    }

    fn release(&self) {
        self.leased.store(false, Ordering::Release);
    }
}

/// A buffer's persistent host mapping. Crate private: writes through it need an idle queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HostPtr(pub(crate) NonNull<u8>);

// SAFETY: Only the stream owning the buffer writes through it.
unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug, Clone)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
    /// The host-access state, present only for a host-visible primary pool.
    #[new(default)]
    pub(crate) host_access: Option<Arc<HostAccess>>,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

/// The memory that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuMemory {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
    /// The host-access state, present only for a host-visible primary pool.
    #[new(default)]
    pub(crate) host_access: Option<Arc<HostAccess>>,
}

impl WgpuResource {
    /// Whether this resource came from a host-visible primary pool.
    pub(crate) fn is_host_visible(&self) -> bool {
        self.host_access.is_some()
    }

    /// Whether a zero-copy read currently holds this allocation's host lease.
    pub(crate) fn is_host_leased(&self) -> bool {
        self.host_access
            .as_ref()
            .is_some_and(|access| access.is_leased())
    }

    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }

    /// The mapping that covers this logical slice, and the offset of the slice
    /// in it. WGPU requires the start to be `MAP_ALIGNMENT`-aligned and the
    /// length to be a multiple of `COPY_BUFFER_ALIGNMENT`.
    fn map_range(&self) -> (Range<u64>, usize) {
        host_mapping_range(self.offset, self.size)
    }

    /// Starts a zero-copy read mapping of this resource.
    ///
    /// The lease is taken before the mapping and stays with the returned
    /// [`HostMapping`] until its [`Bytes`](cubecl_common::bytes::Bytes) drops.
    /// `guard` pins the page the slice lives on, so no new allocation is carved
    /// from it while the host can see the buffer.
    pub(crate) fn begin_host_read(
        &self,
        lease: ManagedMemoryBinding,
        guard: Option<PageGuard>,
        poison: &PoisonWatch,
    ) -> Result<PendingHostRead, HostAccessError> {
        let access = self
            .host_access
            .as_ref()
            .ok_or(HostAccessError::DeviceLocal)?
            .clone();
        access.try_lease()?;

        let (range, view_offset) = self.map_range();
        let (sender, receiver) = cubecl_environment::future::channel::bounded(1);
        poison.wake_on_loss(&sender);
        self.buffer
            .slice(range.clone())
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.try_send(result);
            });

        Ok(PendingHostRead {
            buffer: Some(self.buffer.clone()),
            range,
            view_offset,
            logical_len: self.size as usize,
            lease: Some(lease),
            guard,
            access: Some(access),
            receiver,
        })
    }

    /// Writes `data` into this resource through a mapped buffer.
    ///
    /// This is the whole host-visible write path: a write into an allocation
    /// that is not leased maps in place, no matter what other buffers are doing.
    /// A leased allocation or a failed mapping is an error, never a silent
    /// fallback to staging.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn write_host_visible(
        &self,
        data: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        poison: &PoisonWatch,
    ) -> Result<(), ServerError> {
        let access = self
            .host_access
            .as_ref()
            .ok_or_else(|| ServerError::from(HostAccessError::DeviceLocal))?;
        if data.is_empty() || self.size == 0 {
            return Ok(());
        }
        access
            .try_lease()
            .map_err(|_| ServerError::from(HostAccessError::Leased))?;
        let guard = HostWriteGuard {
            buffer: self.buffer.clone(),
            access: access.clone(),
        };

        let (range, view_offset) = self.map_range();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .slice(range.clone())
            .map_async(wgpu::MapMode::Write, move |result| {
                let _ = sender.send(result);
            });

        // Reuse the device-loss-aware wait, including its probes for a driver
        // that reports a reset only on a later submission.
        poison.wait_unless_lost(device, queue, None)?;
        let result = receiver.recv().map_err(|_| {
            HostAccessError::MapFailed("the write mapping callback was dropped".into())
        })?;
        result.map_err(|err| HostAccessError::MapFailed(err.to_string()))?;

        let mut view = self
            .buffer
            .get_mapped_range_mut(range)
            .map_err(|err| HostAccessError::MapFailed(err.to_string()))?;
        view.slice(view_offset..view_offset + data.len())
            .copy_from_slice(data);
        drop(view);
        drop(guard);
        Ok(())
    }
}

/// A zero-copy read mapping waiting for its asynchronous callback.
pub(crate) struct PendingHostRead {
    buffer: Option<wgpu::Buffer>,
    range: Range<u64>,
    view_offset: usize,
    logical_len: usize,
    lease: Option<ManagedMemoryBinding>,
    guard: Option<PageGuard>,
    access: Option<Arc<HostAccess>>,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl PendingHostRead {
    /// Waits for the mapping callback, then turns the mapped span into the
    /// owner a [`Bytes`](cubecl_common::bytes::Bytes) controller keeps alive.
    pub(crate) async fn complete(mut self) -> Result<HostMapping, ServerError> {
        match self.receiver.recv().await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return Err(HostAccessError::MapFailed(err.to_string()).into());
            }
            Err(_) => {
                return Err(
                    HostAccessError::MapFailed("the host mapping never completed".into()).into(),
                );
            }
        }

        let view = self
            .buffer
            .as_ref()
            .expect("the buffer lives until the callback is awaited")
            .get_mapped_range(self.range.clone())
            .map_err(|err| HostAccessError::MapFailed(err.to_string()))?;
        let buffer = self
            .buffer
            .take()
            .expect("the buffer lives until the callback is awaited");

        Ok(HostMapping {
            buffer,
            view: Some(view),
            view_offset: self.view_offset,
            logical_len: self.logical_len,
            lease: self.lease.take(),
            guard: self.guard.take(),
            access: self.access.take(),
        })
    }
}

impl Drop for PendingHostRead {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            buffer.unmap();
        }
        if let Some(access) = self.access.take() {
            access.release();
        }
    }
}

/// A live zero-copy read mapping, and everything that has to outlive it.
pub(crate) struct HostMapping {
    buffer: wgpu::Buffer,
    view: Option<wgpu::BufferView>,
    view_offset: usize,
    logical_len: usize,
    /// Keeps the allocation out of the pool while the host can see it.
    lease: Option<ManagedMemoryBinding>,
    /// Keeps the rest of a shared page from being carved up under the mapping.
    guard: Option<PageGuard>,
    access: Option<Arc<HostAccess>>,
}

impl Deref for HostMapping {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.view {
            Some(view) => &view[self.view_offset..self.view_offset + self.logical_len],
            None => &[],
        }
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        drop(self.view.take());
        if self.logical_len != 0 {
            self.buffer.unmap();
        }
        if let Some(access) = self.access.take() {
            access.release();
        }
        drop(self.lease.take());
        drop(self.guard.take());
    }
}

/// Releases a mapped write's lease even when a poll or copy returns early.
struct HostWriteGuard {
    buffer: wgpu::Buffer,
    access: Arc<HostAccess>,
}

impl Drop for HostWriteGuard {
    fn drop(&mut self) {
        self.buffer.unmap();
        self.access.release();
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
pub struct WgpuStorage {
    memory: HashMap<StorageId, WgpuMemory>,
    /// The bytes `memory` holds.
    allocated: u64,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    #[allow(unused, reason = "keep it simple")]
    vk_storage: bool,
    /// Whether every buffer this storage allocates is mappable by the host.
    host_visible: bool,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new device-local storage on the given [device](wgpu::Device).
    pub fn new(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
        vk_storage: bool,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
            device,
            buffer_usages: usages,
            mem_alignment,
            vk_storage,
            host_visible: false,
        }
    }

    /// Create a storage whose buffers the host can map, for the primary pool of
    /// a runtime brought up on host-visible memory.
    ///
    /// Every allocation adds `MAP_READ | MAP_WRITE` to `usages`; the device must
    /// have enabled [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] for it.
    pub fn new_with_host_visibility(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
            device,
            buffer_usages: usages,
            mem_alignment,
            // The native Vulkan allocator maps persistently outside wgpu; a
            // host-visible pool maps through wgpu instead.
            vk_storage: false,
            host_visible: true,
        }
    }

    fn allocation_usages(&self) -> BufferUsages {
        if self.host_visible {
            self.buffer_usages | BufferUsages::MAP_READ | BufferUsages::MAP_WRITE
        } else {
            self.buffer_usages
        }
    }

    /// Creates a buffer with the access state its `host_visible` flag calls for.
    fn create_memory(&self, desc: &wgpu::BufferDescriptor<'_>) -> WgpuMemory {
        let buffer = self.device.create_buffer(desc);
        WgpuMemory {
            buffer,
            address: None,
            host_ptr: None,
            host_access: self.host_visible.then(|| Arc::new(HostAccess::default())),
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        let memory = self
            .memory
            .get(&handle.id)
            .ok_or_else(|| IoError::StorageHandleNotFound {
                reason: format!("{} in the wgpu buffer storage", handle.id).into(),
                backtrace: BackTrace::capture(),
            })?;
        Ok(WgpuResource {
            host_ptr: memory.host_ptr,
            host_access: memory.host_access.clone(),
            ..WgpuResource::new(
                memory.buffer.clone(),
                memory.address,
                handle.offset(),
                handle.size(),
            )
        })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let memory = self.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.allocation_usages(),
            mapped_at_creation: false,
        })?;

        self.memory.insert(id, memory);
        self.allocated += alloc_size;
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        if let Some(memory) = self.memory.remove(&id) {
            self.allocated -= memory.buffer.size();
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}

impl WgpuStorage {
    #[cfg(feature = "spirv")]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        if self.vk_storage {
            // wgpu currently doesn't expose this, even though it's used internally for acceleration
            // structures. While we could use the acceleration structure input flag, that would
            // then require ray tracing to be supported. So we need to allocate manually, then import
            // the native buffer into wgpu using `from_raw_managed`.
            // This actually skips some of the buffer batching stuff we don't really want in `gpu_allocator`.
            crate::backend::vulkan::create_storage_buffer(&self.device, desc)
        } else {
            Ok(self.create_memory(desc))
        }
    }

    #[cfg(not(feature = "spirv"))]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        Ok(self.create_memory(desc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_mapping_range_covers_the_logical_slice_in_aligned_bytes() {
        let (range, view_offset) = host_mapping_range(4, 12);

        assert_eq!(view_offset, 4);
        assert_eq!(range, 0..16);
        assert!(range.start.is_multiple_of(wgpu::MAP_ALIGNMENT));
        assert!((range.end - range.start).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
        assert!(range.start + view_offset as u64 <= 4);
        assert!(4 + 12 <= range.end);
    }

    #[test]
    fn a_second_lease_is_refused_until_the_first_is_released() {
        let access = HostAccess::default();

        assert!(access.try_lease().is_ok());
        assert_eq!(access.try_lease(), Err(HostAccessError::AlreadyMapped));
        assert!(access.is_leased());

        access.release();
        assert!(!access.is_leased());
        assert!(access.try_lease().is_ok());
    }
}
