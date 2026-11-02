#[cfg_attr(not(feature = "std"), allow(dead_code))]
// Profiled/ProfileItem are consumed by the std logger only
mod profiling;
pub use profiling::*;

#[cfg(feature = "std")]
mod server;
#[cfg(feature = "std")]
pub use server::*;

#[cfg(not(feature = "std"))]
mod server_no_std;
#[cfg(not(feature = "std"))]
pub use server_no_std::*;
