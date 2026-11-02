//! `f64` math used on hot paths.
//!
//! The crate is `#![no_std]`, but on a hosted target the compiler still gives
//! `f64` its `std` methods; on a bare-metal target it does not, and the
//! workspace's `num-traits` (`libm`) feature supplies them instead. Routing the
//! call sites through these wrappers keeps `std` builds byte-identical and
//! keeps the `no_std` build free of unused imports.

#[cfg(feature = "std")]
pub(crate) fn ln(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
pub(crate) fn ln(x: f64) -> f64 {
    num_traits::Float::ln(x)
}

#[cfg(feature = "std")]
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}

#[cfg(not(feature = "std"))]
pub(crate) fn exp(x: f64) -> f64 {
    num_traits::Float::exp(x)
}

#[cfg(feature = "std")]
pub(crate) fn log(x: f64, base: f64) -> f64 {
    x.log(base)
}

#[cfg(not(feature = "std"))]
pub(crate) fn log(x: f64, base: f64) -> f64 {
    num_traits::Float::log(x, base)
}

#[cfg(feature = "std")]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[cfg(not(feature = "std"))]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    num_traits::Float::powf(x, y)
}

#[cfg(feature = "std")]
pub(crate) fn round(x: f64) -> f64 {
    x.round()
}

#[cfg(not(feature = "std"))]
pub(crate) fn round(x: f64) -> f64 {
    num_traits::Float::round(x)
}

#[cfg(feature = "std")]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
pub(crate) fn floor(x: f64) -> f64 {
    num_traits::Float::floor(x)
}

#[cfg(feature = "std")]
pub(crate) fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
pub(crate) fn ceil(x: f64) -> f64 {
    num_traits::Float::ceil(x)
}
