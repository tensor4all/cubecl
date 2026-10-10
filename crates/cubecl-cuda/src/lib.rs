#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::*;

/// Re-exports for FFI interop with external CUDA libraries.
///
/// These types allow extracting raw CUDA pointers and streams from `CubeCL`'s
/// managed resources, enabling zero-copy interop with cuBLAS, cuSOLVER,
/// cuTENSOR, and other CUDA FFI libraries.
pub mod ffi_interop {
    pub use crate::compute::server::CudaServer;
    pub use crate::compute::storage::gpu::GpuResource;
    pub use crate::compute::stream::Stream;
}

#[cfg(feature = "ptx-wmma")]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::PtxWmmaCompiler;

#[cfg(not(feature = "ptx-wmma"))]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::CudaWmmaCompiler;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f16);
    cubecl_core::testgen_complex_core!();
    cubecl_core::testgen_complex_compare!();
    cubecl_core::testgen_complex_math!();
    cubecl_core::testgen_complex_validation!();

    /// A producer failure must be observable without flushing: `flush` also retires staged host
    /// bytes and arms a fence that the next submission synchronizes, so a host-side publisher that
    /// only needs the error (and not a device wait) uses `check_errors` instead.
    #[test]
    fn client_check_errors_reports_a_failed_launch_without_flushing() {
        use cubecl_core::prelude::*;
        use cubecl_core::runtime_tests::launch::kernel_resource_errors;
        use cubecl_runtime::server::{LaunchError, ServerError};

        let client = TestRuntime::client(&Default::default());
        let max_shared_size = client.properties().hardware.max_shared_memory_size;
        let shared_size = (max_shared_size + 1).div_ceil(size_of::<u32>());
        let handle = client.create_from_slice(u32::as_bytes(&[0]));

        kernel_resource_errors::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(handle.clone(), 1) },
            shared_size,
        );

        match client.check_errors() {
            Err(ServerError::ServerUnhealthy { mut errors, .. }) => {
                let error = errors.remove(0);
                assert!(
                    matches!(error, ServerError::Launch(LaunchError::TooManyResources(_))),
                    "expected a reported launch failure, got {error:?}"
                );
            }
            other => panic!("expected a reported launch failure, got {other:?}"),
        }

        // The error queue was drained, so the stream is healthy again without any device wait.
        client
            .check_errors()
            .expect("a stream whose errors were drained reports none");
    }
}
