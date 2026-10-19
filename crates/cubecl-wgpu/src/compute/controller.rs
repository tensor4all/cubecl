use crate::compute::storage::HostMapping;
use core::mem::MaybeUninit;
use cubecl_common::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, Reader, Writer,
};
use cubecl_server::memory_management::ManagedMemoryBinding;
use wgpu::BufferView;

/// Controller for managing wgpu staging buffers managed by a memory pool.
pub struct WgpuAllocController {
    view: Option<BufferView>,
    buffer: wgpu::Buffer,
    _binding: ManagedMemoryBinding,
}

impl Drop for WgpuAllocController {
    fn drop(&mut self) {
        // Drop the view first, then unmap the buffer.
        // This ensures proper cleanup order since the view borrows from the buffer.
        drop(self.view.take());
        // We unmap the buffer and release the binding so that the same buffer can be used again.
        self.buffer.unmap();
    }
}

impl AllocationController for WgpuAllocController {
    fn alloc_align(&self) -> usize {
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    // A mapped wgpu staging buffer is always host-resident: the policy never forces a copy here.
    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        // - Caller promises not to write uninitialized values.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(bytes.as_ptr() as *mut MaybeUninit<u8>, bytes.len())
        })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[std::mem::MaybeUninit<u8>], AccessError> {
        let bytes: &[u8] = self.view.as_ref().unwrap();
        // SAFETY:
        // - MaybeUninit<u8> has the same layout as u8
        Ok(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const MaybeUninit<u8>, bytes.len())
        })
    }
}

impl WgpuAllocController {
    /// Creates a new allocation controller for a managed wgpu staging buffer.
    ///
    /// # Arguments
    ///
    /// * `binding` - The memory binding for the managed buffer.
    /// * `buffer` - The wgpu buffer.
    ///
    /// # Returns
    ///
    /// The controller.
    pub fn init(binding: ManagedMemoryBinding, buffer: wgpu::Buffer) -> Self {
        // Needs immutable as of wgpu v29, mutable doesn't allow dereferencing as slice.
        // This only affects wgpu's internal overlap checks so it should be fine as long as we
        // map the whole buffer anyways.
        let buf_view = buffer.get_mapped_range(..).unwrap();

        Self {
            view: Some(buf_view),
            buffer,
            _binding: binding,
        }
    }
}

/// A [`Bytes`] backed by a host-visible primary buffer's own mapping.
///
/// Reads are zero-copy: the controller derefs into the mapped device allocation.
/// The mapping keeps the allocation leased and its page guarded, so the buffer
/// cannot be written by a launch or returned to the pool while the [`Bytes`]
/// lives. Mutable access copies the mapped view into a private buffer on first
/// write, so the primary allocation is never modified through a read.
pub(crate) struct WgpuMappedReadController {
    mapping: HostMapping,
    /// The private buffer a mutable access copied into, if one happened.
    copy: Option<Bytes>,
    len: usize,
}

impl WgpuMappedReadController {
    pub(crate) fn new(mapping: HostMapping, len: usize) -> Self {
        Self {
            mapping,
            copy: None,
            len,
        }
    }

    fn mapped(&self) -> &[u8] {
        &self.mapping
    }
}

impl AllocationController for WgpuMappedReadController {
    fn alloc_align(&self) -> usize {
        // The logical slice starts at a WGPU mapping alignment, which is what the
        // staging read path reports too. Reporting the host allocation's wider
        // alignment would make a downstream `try_into_vec::<u32>` refuse it.
        wgpu::COPY_BUFFER_ALIGNMENT as usize
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    fn capacity(&self) -> usize {
        self.len
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        let bytes = match &self.copy {
            Some(copy) => copy
                .read(Reader::new())
                .map_err(|err| AccessError::Read(err.to_string()))?,
            None => self.mapped(),
        };
        // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`, and every byte of
        // the mapped logical slice is initialized.
        Ok(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const MaybeUninit<u8>, bytes.len())
        })
    }

    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        if self.copy.is_none() {
            if !policy.copy_allowed() {
                return Err(AccessError::WouldCopy);
            }
            self.copy = Some(Bytes::from_bytes_vec(self.mapped().to_vec()));
        }
        let bytes = self
            .copy
            .as_mut()
            .expect("the copy was just created")
            .write(Writer::new())
            .map_err(|err| AccessError::Read(err.to_string()))?;
        // SAFETY: the private copy is fully initialized, and the caller promises
        // to write only initialized data.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut MaybeUninit<u8>, bytes.len())
        })
    }
}
