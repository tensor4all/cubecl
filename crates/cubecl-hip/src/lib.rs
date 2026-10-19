#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compiler;
pub mod compute;
pub mod device;
pub mod runtime;
pub use compiler::{HipCompilationOptions, HipCompiler, HipRepresentation};
pub use device::*;
pub use runtime::HipRuntime;

// HIP has no runtime on Apple platforms. Keep the compiler-only unit tests above
// available there, but don't generate hundreds of tests that unconditionally
// open device 0.
#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use half::{bf16, f16};
    pub type TestRuntime = crate::HipRuntime;

    cubecl_std::testgen!();
    cubecl_core::testgen_all!(f32: [f16, bf16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_core::testgen_complex_validation!();
    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_core::testgen_profiling!();
    cubecl_core::testgen_profiling_nested!();

    fn external_op_copy(
        stream: cubecl_hip_sys::hipStream_t,
        src: u64,
        dst: u64,
        len: u64,
    ) -> Result<(), cubecl_core::server::ServerError> {
        // SAFETY: the test keeps both allocations alive until the stream has
        // run the copy and the read that follows it.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpyDtoDAsync(
                dst as *mut core::ffi::c_void,
                src as *mut core::ffi::c_void,
                len as usize,
                stream,
            )
        };
        cubecl_server::driver::checked("hipMemcpyDtoDAsync", status)?;
        Ok(())
    }

    fn external_op_fill(
        stream: cubecl_hip_sys::hipStream_t,
        dst: u64,
        len: u64,
        value: u32,
    ) -> Result<(), cubecl_core::server::ServerError> {
        // SAFETY: the test fills a live device allocation of `len` bytes.
        let status = unsafe {
            cubecl_hip_sys::hipMemsetD32Async(
                dst as *mut core::ffi::c_void,
                value as core::ffi::c_int,
                len as usize / core::mem::size_of::<u32>(),
                stream,
            )
        };
        cubecl_server::driver::checked("hipMemsetD32Async", status)?;
        Ok(())
    }

    cubecl_core::testgen_external_op!(external_op_copy, external_op_fill, true);
}
