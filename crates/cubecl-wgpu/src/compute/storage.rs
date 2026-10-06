use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::{IoError, ServerError};
use cubecl_runtime::{
    memory_management::ManagedMemoryBinding,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use hashbrown::HashMap;
use std::{
    num::NonZeroU64,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

const ACCESS_GPU_IDLE: usize = 0;
const ACCESS_HOST_MAPPED: usize = 1 << (usize::BITS - 1);
const ACCESS_GPU_COUNT_MASK: usize = ACCESS_HOST_MAPPED - 1;

static NEXT_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

/// Error returned when host and GPU access to a WGPU allocation cannot be coordinated.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostAccessError {
    /// The allocation was created without host-visible primary memory enabled.
    DeviceLocalAllocation,
    /// Another host mapping is already active for the physical allocation.
    OverlappingHostMapping,
    /// GPU work has reserved the allocation and has not yet been submitted.
    GpuAccessInProgress,
    /// GPU use was requested while a host mapping is active.
    MappedForHost,
    /// WGPU failed to complete the asynchronous map callback.
    MapCallbackFailure(String),
    /// WGPU failed while polling the device for mapping completion.
    DevicePollFailure(String),
}

impl core::fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DeviceLocalAllocation => {
                f.write_str("the WGPU allocation is device-local and cannot be mapped by the host")
            }
            Self::OverlappingHostMapping => {
                f.write_str("the WGPU allocation already has an active host mapping")
            }
            Self::GpuAccessInProgress => f.write_str(
                "the WGPU allocation has GPU work pending submission and cannot be mapped by the host",
            ),
            Self::MappedForHost => f.write_str(
                "the WGPU allocation is mapped for host access and cannot be used by the GPU",
            ),
            Self::MapCallbackFailure(reason) => {
                write!(f, "the WGPU map callback failed: {reason}")
            }
            Self::DevicePollFailure(reason) => {
                write!(
                    f,
                    "polling the WGPU device for host mapping failed: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for HostAccessError {}

impl From<HostAccessError> for ServerError {
    fn from(error: HostAccessError) -> Self {
        Self::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

#[derive(Debug)]
struct AllocationAccessInner {
    allocation_id: u64,
    state: AtomicUsize,
}

#[derive(Clone, Debug)]
struct AllocationAccess {
    inner: Arc<AllocationAccessInner>,
}

impl AllocationAccess {
    // `fetch_update` is deprecated in favour of `try_update` on current stable,
    // which postdates the crate's MSRV.
    #[allow(deprecated)]
    fn new() -> Self {
        let allocation_id = NEXT_ALLOCATION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("WGPU allocation ID overflowed");

        Self {
            inner: Arc::new(AllocationAccessInner {
                allocation_id,
                state: AtomicUsize::new(ACCESS_GPU_IDLE),
            }),
        }
    }

    fn allocation_id(&self) -> u64 {
        self.inner.allocation_id
    }

    fn acquire_host(&self) -> Result<HostAccessToken, HostAccessError> {
        self.inner
            .state
            .compare_exchange(
                ACCESS_GPU_IDLE,
                ACCESS_HOST_MAPPED,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map_err(|state| {
                if state == ACCESS_HOST_MAPPED {
                    HostAccessError::OverlappingHostMapping
                } else {
                    HostAccessError::GpuAccessInProgress
                }
            })?;

        Ok(HostAccessToken {
            access: Some(self.inner.clone()),
        })
    }

    fn ensure_gpu_access(&self) -> Result<(), HostAccessError> {
        if self.inner.state.load(Ordering::Acquire) == ACCESS_HOST_MAPPED {
            Err(HostAccessError::MappedForHost)
        } else {
            Ok(())
        }
    }

    fn acquire_gpu(&self) -> Result<GpuAccessToken, HostAccessError> {
        let mut state = self.inner.state.load(Ordering::Acquire);
        loop {
            if state == ACCESS_HOST_MAPPED {
                return Err(HostAccessError::MappedForHost);
            }
            assert_ne!(
                state, ACCESS_GPU_COUNT_MASK,
                "too many simultaneous WGPU GPU reservations"
            );
            match self.inner.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(GpuAccessToken {
                        access: Some(self.inner.clone()),
                        lease: None,
                    });
                }
                Err(current) => state = current,
            }
        }
    }
}

/// RAII reservation preventing host mapping until queued GPU use has been submitted.
#[doc(hidden)]
#[derive(Debug)]
pub struct GpuAccessToken {
    access: Option<Arc<AllocationAccessInner>>,
    lease: Option<ManagedMemoryBinding>,
}

impl Drop for GpuAccessToken {
    fn drop(&mut self) {
        if let Some(access) = self.access.take() {
            let previous = access.state.fetch_sub(1, Ordering::Release);
            debug_assert!(previous > 0 && previous < ACCESS_HOST_MAPPED);
        }
    }
}

#[derive(Debug)]
struct HostAccessToken {
    access: Option<Arc<AllocationAccessInner>>,
}

impl Drop for HostAccessToken {
    fn drop(&mut self) {
        if let Some(access) = self.access.take() {
            access.state.store(ACCESS_GPU_IDLE, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct WgpuAllocation {
    buffer: wgpu::Buffer,
    device: Option<wgpu::Device>,
    allocation_access: AllocationAccess,
    host_visible: bool,
}

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, Arc<WgpuAllocation>>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    host_visible: bool,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
///
/// For [`crate::WgpuRuntime`], callers can resolve a public `CubeCL` allocation handle with
/// [`cubecl_runtime::client::ComputeClient::get_resource`]. Call the returned
/// [`cubecl_runtime::storage::ManagedResource`]'s `resource` method to borrow this value, and then
/// use [`Self::allocation_id`], [`Self::map_read`], or [`Self::map_write`]. Managed resources,
/// cloned `WgpuResource` values, mapped guards, and queued GPU reservations each retain the
/// memory-manager lease for as long as they can access the allocation.
#[derive(Clone, Debug)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
    allocation: Arc<WgpuAllocation>,
    lease: Option<ManagedMemoryBinding>,
}

impl WgpuResource {
    /// Creates a device-local resource from an existing WGPU buffer.
    ///
    /// Resources allocated by [`WgpuStorage`] additionally retain their device so they can opt in
    /// to synchronous host mapping.
    pub fn new(buffer: wgpu::Buffer, offset: u64, size: u64) -> Self {
        let allocation = Arc::new(WgpuAllocation {
            buffer: buffer.clone(),
            device: None,
            allocation_access: AllocationAccess::new(),
            host_visible: false,
        });

        Self {
            buffer,
            offset,
            size,
            allocation,
            lease: None,
        }
    }

    fn from_allocation(allocation: Arc<WgpuAllocation>, offset: u64, size: u64) -> Self {
        Self {
            buffer: allocation.buffer.clone(),
            offset,
            size,
            allocation,
            lease: None,
        }
    }

    pub(crate) fn with_lease(mut self, lease: ManagedMemoryBinding) -> Self {
        self.lease = Some(lease);
        self
    }

    /// Returns the process-unique identity of the underlying physical allocation.
    pub fn allocation_id(&self) -> u64 {
        self.allocation.allocation_access.allocation_id()
    }

    /// Validates that the resource can currently be used by the GPU.
    pub fn ensure_gpu_access(&self) -> Result<(), HostAccessError> {
        self.allocation.allocation_access.ensure_gpu_access()
    }

    pub(crate) fn acquire_gpu(&self) -> Result<GpuAccessToken, HostAccessError> {
        let mut token = self.allocation.allocation_access.acquire_gpu()?;
        token.lease = self.lease.clone();
        Ok(token)
    }

    fn acquire_host(&self) -> Result<HostAccessToken, HostAccessError> {
        if !self.allocation.host_visible {
            return Err(HostAccessError::DeviceLocalAllocation);
        }
        self.allocation.allocation_access.acquire_host()
    }

    #[cfg(not(target_family = "wasm"))]
    fn map_range(&self) -> core::ops::Range<u64> {
        let end = self.offset + self.size;
        self.offset..end.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
    }

    /// Maps the exact logical resource range for synchronous host reads.
    #[cfg(not(target_family = "wasm"))]
    pub fn map_read(&self) -> Result<WgpuMappedReadGuard, HostAccessError> {
        let token = self.acquire_host()?;

        if self.size == 0 {
            return Ok(WgpuMappedReadGuard {
                allocation: self.allocation.clone(),
                view: None,
                logical_len: 0,
                token: Some(token),
                _lease: self.lease.clone(),
            });
        }

        let range = self.map_range();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .map_async(wgpu::MapMode::Read, range.clone(), move |result| {
                let _ = sender.send(result);
            });

        let device = self
            .allocation
            .device
            .as_ref()
            .expect("host-visible WGPU allocations always retain their device");
        if let Err(err) = device.poll(wgpu::PollType::wait_indefinitely()) {
            self.buffer.unmap();
            return Err(HostAccessError::DevicePollFailure(err.to_string()));
        }

        let callback_result = match receiver.recv() {
            Ok(result) => result,
            Err(err) => {
                self.buffer.unmap();
                return Err(HostAccessError::MapCallbackFailure(err.to_string()));
            }
        };
        if let Err(err) = callback_result {
            self.buffer.unmap();
            return Err(HostAccessError::MapCallbackFailure(err.to_string()));
        }

        let view = self.buffer.get_mapped_range(range);
        Ok(WgpuMappedReadGuard {
            allocation: self.allocation.clone(),
            view: Some(view),
            logical_len: self.size as usize,
            token: Some(token),
            _lease: self.lease.clone(),
        })
    }

    /// Maps the exact logical resource range for synchronous host writes.
    #[cfg(not(target_family = "wasm"))]
    pub fn map_write(&self) -> Result<WgpuMappedWriteGuard, HostAccessError> {
        let token = self.acquire_host()?;

        if self.size == 0 {
            return Ok(WgpuMappedWriteGuard {
                allocation: self.allocation.clone(),
                view: None,
                logical_len: 0,
                token: Some(token),
                _lease: self.lease.clone(),
            });
        }

        let range = self.map_range();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .map_async(wgpu::MapMode::Write, range.clone(), move |result| {
                let _ = sender.send(result);
            });

        let device = self
            .allocation
            .device
            .as_ref()
            .expect("host-visible WGPU allocations always retain their device");
        if let Err(err) = device.poll(wgpu::PollType::wait_indefinitely()) {
            self.buffer.unmap();
            return Err(HostAccessError::DevicePollFailure(err.to_string()));
        }

        let callback_result = match receiver.recv() {
            Ok(result) => result,
            Err(err) => {
                self.buffer.unmap();
                return Err(HostAccessError::MapCallbackFailure(err.to_string()));
            }
        };
        if let Err(err) = callback_result {
            self.buffer.unmap();
            return Err(HostAccessError::MapCallbackFailure(err.to_string()));
        }

        let view = self.buffer.get_mapped_range_mut(range);
        Ok(WgpuMappedWriteGuard {
            allocation: self.allocation.clone(),
            view: Some(view),
            logical_len: self.size as usize,
            token: Some(token),
            _lease: self.lease.clone(),
        })
    }

    /// Synchronous mapping is unavailable on WebAssembly because `Device::poll` cannot block.
    #[cfg(target_family = "wasm")]
    pub fn map_read(&self) -> Result<WgpuMappedReadGuard, HostAccessError> {
        Err(HostAccessError::DevicePollFailure(
            "synchronous host mapping is unsupported on WebAssembly".to_string(),
        ))
    }

    /// Synchronous mapping is unavailable on WebAssembly because `Device::poll` cannot block.
    #[cfg(target_family = "wasm")]
    pub fn map_write(&self) -> Result<WgpuMappedWriteGuard, HostAccessError> {
        Err(HostAccessError::DevicePollFailure(
            "synchronous host mapping is unsupported on WebAssembly".to_string(),
        ))
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

    /// Returns the binding view after checking that no host mapping is active.
    pub fn try_as_wgpu_bind_resource(&self) -> Result<wgpu::BindingResource<'_>, HostAccessError> {
        self.ensure_gpu_access()?;
        Ok(self.as_wgpu_bind_resource())
    }
}

/// Owning RAII guard for a synchronously mapped host-readable WGPU resource.
#[derive(Debug)]
pub struct WgpuMappedReadGuard {
    allocation: Arc<WgpuAllocation>,
    view: Option<wgpu::BufferView>,
    logical_len: usize,
    token: Option<HostAccessToken>,
    _lease: Option<ManagedMemoryBinding>,
}

impl Deref for WgpuMappedReadGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.view {
            Some(view) => &view[..self.logical_len],
            None => &[],
        }
    }
}

impl Drop for WgpuMappedReadGuard {
    fn drop(&mut self) {
        drop(self.view.take());
        if self.logical_len != 0 {
            self.allocation.buffer.unmap();
        }
        drop(self.token.take());
    }
}

/// Owning RAII guard for a synchronously mapped host-writable WGPU resource.
///
/// WGPU write mappings may be write-combining memory, so wgpu 29 intentionally exposes them as
/// [`wgpu::WriteOnly`] rather than `&mut [u8]`. Use [`Self::as_write_only`] for zero-copy writes.
#[derive(Debug)]
pub struct WgpuMappedWriteGuard {
    allocation: Arc<WgpuAllocation>,
    view: Option<wgpu::BufferViewMut>,
    logical_len: usize,
    token: Option<HostAccessToken>,
    _lease: Option<ManagedMemoryBinding>,
}

impl WgpuMappedWriteGuard {
    /// Returns a zero-copy write-only view of the exact logical resource range.
    pub fn as_write_only(&mut self) -> Option<wgpu::WriteOnly<'_, [u8]>> {
        self.view
            .as_mut()
            .map(|view| view.slice(..self.logical_len))
    }

    /// Copies bytes directly into the mapped logical resource range.
    pub fn copy_from_slice(&mut self, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.logical_len);
        if let Some(mut view) = self.as_write_only() {
            view.copy_from_slice(bytes);
        }
    }

    /// Returns the logical resource length in bytes.
    pub fn len(&self) -> usize {
        self.logical_len
    }

    /// Returns whether the logical resource is empty.
    pub fn is_empty(&self) -> bool {
        self.logical_len == 0
    }
}

impl Drop for WgpuMappedWriteGuard {
    fn drop(&mut self) {
        drop(self.view.take());
        if self.logical_len != 0 {
            self.allocation.buffer.unmap();
        }
        drop(self.token.take());
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(mem_alignment: usize, device: wgpu::Device, usages: BufferUsages) -> Self {
        Self::new_with_host_visibility(mem_alignment, device, usages, false)
    }

    pub(crate) fn new_with_host_visibility(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
        host_visible: bool,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
            host_visible,
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let allocation = self.memory.get(&handle.id).unwrap();
        WgpuResource::from_allocation(allocation.clone(), handle.offset(), handle.size())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        });

        self.memory.insert(
            id,
            Arc::new(WgpuAllocation {
                buffer,
                device: Some(self.device.clone()),
                allocation_access: AllocationAccess::new(),
                host_visible: self.host_visible,
            }),
        );
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_allocation_access_retains_nonzero_identity() {
        let access = Arc::new(AllocationAccess::new());
        let cloned = access.clone();

        assert_ne!(access.allocation_id(), 0);
        assert_eq!(access.allocation_id(), cloned.allocation_id());
    }

    #[test]
    fn host_access_excludes_other_host_and_gpu_access_until_drop() {
        let access = AllocationAccess::new();
        let token = access.acquire_host().unwrap();

        assert_eq!(
            access.acquire_host().unwrap_err(),
            HostAccessError::OverlappingHostMapping
        );
        assert_eq!(
            access.ensure_gpu_access(),
            Err(HostAccessError::MappedForHost)
        );

        drop(token);

        assert_eq!(access.ensure_gpu_access(), Ok(()));
        assert!(access.acquire_host().is_ok());
    }

    #[test]
    fn gpu_reservation_excludes_host_access_until_submission() {
        let access = AllocationAccess::new();
        let worker_access = access.clone();
        let (reserved_tx, reserved_rx) = std::sync::mpsc::sync_channel(1);
        let (submitted_tx, submitted_rx) = std::sync::mpsc::sync_channel(1);

        let worker = std::thread::spawn(move || {
            let reservation = worker_access.acquire_gpu().unwrap();
            reserved_tx.send(()).unwrap();
            submitted_rx.recv().unwrap();
            drop(reservation);
        });

        reserved_rx.recv().unwrap();
        assert_eq!(
            access.acquire_host().unwrap_err(),
            HostAccessError::GpuAccessInProgress
        );

        submitted_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(access.acquire_host().is_ok());
    }
}
