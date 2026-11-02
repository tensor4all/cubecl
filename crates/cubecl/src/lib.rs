#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

#[cfg(feature = "wgpu")]
pub use cubecl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use cubecl_cuda as cuda;

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;

#[cfg(test_runtime_default)]
pub type TestRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(test_runtime_wgpu)]
pub type TestRuntime = wgpu::WgpuRuntime;

#[cfg(test_runtime_cuda)]
pub type TestRuntime = cuda::CudaRuntime;
