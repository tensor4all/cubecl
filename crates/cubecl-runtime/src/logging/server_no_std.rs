use alloc::string::String;
use core::fmt::Display;

use cubecl_common::profile::ProfileDuration;

use super::ProfileLevel;
use crate::config::memory::MemoryLogLevel;
use crate::config::streaming::StreamingLogLevel;

/// Server logger for targets without `std`.
///
/// The `std` version owns a channel and a detached task that drains it. Neither
/// exists here: `spawn_detached_fut` panics without `std` or WASM, and the
/// channel backend needs `std` too. Every level is therefore fixed to disabled,
/// so callers keep the same API and pay no formatting cost on the hot path: a
/// `ServerLogger` without `std` never logs.
#[derive(Debug, Default)]
pub struct ServerLogger {
    profile_level: Option<ProfileLevel>,
    log_compile_info: bool,
}

impl ServerLogger {
    /// Returns the profile level, none if profiling is deactivated.
    ///
    /// Always `None` without `std`.
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        self.profile_level
    }

    /// Returns true if compilation info should be logged.
    ///
    /// Always `false` without `std`.
    pub fn compilation_activated(&self) -> bool {
        self.log_compile_info
    }

    /// No-op without `std`; see the type documentation.
    pub fn log_compilation<I>(&self, _arg: &I)
    where
        I: Display,
    {
    }

    /// No-op without `std`; see the type documentation.
    pub fn log_streaming<I: FnOnce() -> String, C: FnOnce(StreamingLogLevel) -> bool>(
        &self,
        _cond: C,
        _format: I,
    ) {
    }

    /// No-op without `std`; see the type documentation.
    pub fn log_memory<I: FnOnce() -> String, C: FnOnce(MemoryLogLevel) -> bool>(
        &self,
        _cond: C,
        _format: I,
    ) {
    }

    /// No-op without `std`; see the type documentation.
    pub fn register_execution(&self, _name: impl Display) {}

    /// No-op without `std`; see the type documentation.
    pub fn register_profiled(&self, _name: impl Display, _duration: ProfileDuration) {}

    /// No-op without `std`; see the type documentation.
    pub fn profile_summary(&self) {}
}
