//! The stream work a device has submitted and no [`flush`](crate::client::Client::flush)
//! has pushed yet, which every client of the device reads without reaching its thread.
//!
//! The submissions are counted by [`Client`](crate::client::Client) on the caller's
//! thread, right after it handed the task to the device; only a flush retires them. That
//! direction is what makes the answer usable: a caller returning from a submission has
//! already published its mark, so a later question from any clone of any client sees it,
//! while a flush can only ever retire what was published before it went out.
//!
//! A record the server maintained instead could not have both: the caller would have to
//! ask the device thread what it submitted, which is the round trip the query exists to
//! avoid.

use core::sync::atomic::{AtomicU64, Ordering};

use cubecl_environment::collections::HashMap;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{Arc, RwLock};

/// What each stream of a device has submitted and what its last flush pushed: the record
/// [`Client::has_pending_work`](crate::client::Client::has_pending_work) answers from.
///
/// Every client of the device shares one record through the
/// [server utilities](crate::server::ServerUtilities), so the answer for a stream covers
/// the work of every clone of every client on it, not only the one that asks.
///
/// Entries are never removed and counters never reset. Reusing them would let a count
/// mean two different submissions, and an older flush then retire work it never pushed —
/// exactly the recycling that makes a caller-side record unsound.
#[derive(Debug, Clone)]
pub(crate) struct PendingWork(Arc<PendingRecord>);

/// What a flush takes before it enqueues: the counts it will have pushed once it returns.
///
/// Only a flush reads it, so its fields stay private to this module; a caller holds one
/// between [`snapshot`](PendingWork::snapshot) and [`retire`](PendingWork::retire).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Snapshot {
    submitted: u64,
    epoch: u64,
}

/// Marks a stream — or, without one, every stream — when it drops.
///
/// A mark published in `Drop` holds after the submission both when it returned and when it
/// unwound, and never before it: the one placement that can neither be retired by a flush
/// that did not push the task nor be skipped by a panic.
#[must_use = "the mark is published when the guard drops, so it has to outlive the submit"]
pub(crate) struct MarkOnExit<'a> {
    pending: &'a PendingWork,
    stream: Option<StreamId>,
}

impl Drop for MarkOnExit<'_> {
    fn drop(&mut self) {
        match self.stream {
            Some(stream) => self.pending.mark(stream),
            None => self.pending.mark_all(),
        }
    }
}

/// One logical stream's counts: how much it submitted and how much a flush has pushed.
#[derive(Debug, Default)]
struct StreamWork {
    /// How many tasks of this stream were handed to the device.
    submitted: AtomicU64,
    /// The highest [`submitted`](Self::submitted) a successful flush of this stream pushed.
    retired: AtomicU64,
    /// The highest [`PendingRecord::epoch`] a successful flush of this stream pushed.
    retired_epoch: AtomicU64,
}

/// The streams of one device, plus the service-wide epoch a closure on the server invalidates.
#[derive(Debug)]
struct PendingRecord {
    /// One entry per stream the record has seen, created on first use — a mark or a flush's
    /// snapshot — and kept.
    streams: RwLock<HashMap<StreamId, StreamWork>>,
    /// How many times a [`with_server`](crate::client::Client::with_server) closure has run.
    ///
    /// Such a closure runs on the device thread with the server to itself and may submit
    /// work on any stream, which no per-stream mark in the client saw. Bumping one count
    /// for the whole service makes every stream pending until its own flush retires an
    /// epoch taken before that flush; a stream the record never saw is pending for the
    /// same reason, since a closure may have been its first user.
    epoch: AtomicU64,
}

impl Default for PendingWork {
    fn default() -> Self {
        Self(Arc::new(PendingRecord {
            streams: RwLock::new(HashMap::default()),
            epoch: AtomicU64::new(0),
        }))
    }
}

impl PendingWork {
    /// The record of a device that has submitted nothing yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record that one task of `stream` was handed to the device.
    ///
    /// Call it after the submit returned, never before: a count published while the task
    /// is still on its way could be retired by a flush that never pushed it.
    pub(crate) fn mark(&self, stream: StreamId) {
        self.with_entry(stream, |work| {
            work.submitted.fetch_add(1, Ordering::Release);
        });
    }

    /// Record that a `with_server` closure ran: it may have submitted work on any stream,
    /// so every stream counts as pending until its own flush pushes that too.
    pub(crate) fn mark_all(&self) {
        self.0.epoch.fetch_add(1, Ordering::Release);
    }

    /// A guard that [marks](Self::mark) `stream` when it drops: take it before the submit,
    /// and the mark is published once the submitting call is over — whether the submit
    /// returned or unwound, since a task that panicked may still have queued part of its
    /// work.
    pub(crate) fn mark_on_exit(&self, stream: StreamId) -> MarkOnExit<'_> {
        MarkOnExit {
            pending: self,
            stream: Some(stream),
        }
    }

    /// [`mark_on_exit`](Self::mark_on_exit) for a submission that may touch any stream:
    /// it [bumps the epoch](Self::mark_all) instead.
    pub(crate) fn mark_all_on_exit(&self) -> MarkOnExit<'_> {
        MarkOnExit {
            pending: self,
            stream: None,
        }
    }

    /// The counts a flush of `stream` is about to push: take it before the flush is
    /// enqueued, so everything it counted is already ahead of the flush in the queue.
    pub(crate) fn snapshot(&self, stream: StreamId) -> Snapshot {
        self.with_entry(stream, |work| Snapshot {
            submitted: work.submitted.load(Ordering::Acquire),
            epoch: self.0.epoch.load(Ordering::Acquire),
        })
    }

    /// Record that the flush which took `snapshot` pushed everything it counted.
    ///
    /// Only a successful flush retires: a failed one pushed nothing past the work it
    /// counted, so the stream stays pending for the next attempt to report again.
    pub(crate) fn retire(&self, stream: StreamId, snapshot: Snapshot) {
        // `fetch_max`, not a store: flushes of one stream can overlap, and the older of
        // two must not move the record back behind what the newer one already pushed.
        self.with_entry(stream, |work| {
            work.retired
                .fetch_max(snapshot.submitted, Ordering::Release);
            work.retired_epoch
                .fetch_max(snapshot.epoch, Ordering::Release);
        });
    }

    /// Whether `stream` may hold work that a flush has not pushed yet.
    ///
    /// Work submitted after this load is concurrent with the question and not covered;
    /// work submitted before it is, whichever thread or clone submitted it.
    pub(crate) fn is_pending(&self, stream: StreamId) -> bool {
        let streams = self.0.streams.read();
        let Some(work) = streams.get(&stream) else {
            // No task of this stream was ever counted here, but a `with_server` closure
            // may still have used it: the epoch is all the record can know about it.
            return self.0.epoch.load(Ordering::Acquire) > 0;
        };

        // What was submitted is read before what retired it — the order a flush takes its
        // snapshot in — so the retire side is never older than the submission side it is
        // compared against, and a stream that looks caught up really had a flush taken
        // after every count it shows. A mark published while these loads run is concurrent
        // with this query, and the answer does not cover it either way.
        let (submitted, epoch) = (
            work.submitted.load(Ordering::Acquire),
            self.0.epoch.load(Ordering::Acquire),
        );
        let (retired, retired_epoch) = (
            work.retired.load(Ordering::Acquire),
            work.retired_epoch.load(Ordering::Acquire),
        );

        submitted > retired || epoch > retired_epoch
    }

    /// Run `f` on the record of `stream`, inserted on first use. Entries are never removed,
    /// so a stream is inserted at most once and the write lock is only taken then; every
    /// other call works under the read lock, on atomics alone.
    fn with_entry<R>(&self, stream: StreamId, f: impl FnOnce(&StreamWork) -> R) -> R {
        {
            let streams = self.0.streams.read();
            if let Some(work) = streams.get(&stream) {
                return f(work);
            }
        }

        f(self.0.streams.write().entry(stream).or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: StreamId = StreamId { value: 1 };
    const OTHER: StreamId = StreamId { value: 2 };

    /// A stream nobody touched is not pending, and one task makes it pending until a flush
    /// taken afterwards retires it. This is the whole contract the client documents.
    #[test]
    fn a_mark_is_pending_until_a_later_flush_retires_it() {
        let pending = PendingWork::new();
        assert!(!pending.is_pending(STREAM), "a fresh device holds nothing");

        pending.mark(STREAM);
        assert!(pending.is_pending(STREAM));

        let snapshot = pending.snapshot(STREAM);
        pending.retire(STREAM, snapshot);
        assert!(!pending.is_pending(STREAM));
    }

    /// A snapshot taken before a mark does not retire it: the task was not in the queue
    /// when that flush went out, so retiring it would claim the flush pushed it.
    #[test]
    fn a_snapshot_taken_before_a_mark_does_not_retire_it() {
        let pending = PendingWork::new();

        let snapshot = pending.snapshot(STREAM);
        pending.mark(STREAM);
        pending.retire(STREAM, snapshot);

        assert!(
            pending.is_pending(STREAM),
            "the flush never pushed the task marked after its snapshot"
        );
    }

    /// Two flushes of one stream can overlap, and the older one returning last must not
    /// move the record back behind what the newer one already pushed.
    #[test]
    fn retiring_an_older_snapshot_after_a_newer_one_does_not_regress() {
        let pending = PendingWork::new();

        pending.mark(STREAM);
        let older = pending.snapshot(STREAM);
        pending.mark(STREAM);
        let newer = pending.snapshot(STREAM);

        pending.retire(STREAM, newer);
        pending.retire(STREAM, older);

        assert!(
            !pending.is_pending(STREAM),
            "the newer flush pushed the second task; the older retire must not undo it"
        );
    }

    /// A `with_server` closure may touch any stream, so it makes every stream pending,
    /// including one the record never saw, until each stream's own flush retires an epoch
    /// taken before that flush.
    #[test]
    fn a_with_server_closure_makes_every_stream_pending_until_its_own_flush() {
        let pending = PendingWork::new();

        pending.mark_all();
        assert!(
            pending.is_pending(STREAM),
            "an unknown stream may have been the closure's first user"
        );
        let snapshot = pending.snapshot(STREAM);
        pending.retire(STREAM, snapshot);
        assert!(!pending.is_pending(STREAM));

        // A stream that had pushed everything it submitted goes pending again, because the
        // closure may have submitted more on it.
        pending.mark(STREAM);
        let snapshot = pending.snapshot(STREAM);
        pending.retire(STREAM, snapshot);
        assert!(!pending.is_pending(STREAM));

        pending.mark_all();
        assert!(pending.is_pending(STREAM));

        // Another stream's flush retires its own epoch snapshot and nothing else's.
        let snapshot = pending.snapshot(OTHER);
        pending.retire(OTHER, snapshot);
        assert!(!pending.is_pending(OTHER));
        assert!(
            pending.is_pending(STREAM),
            "flushing a neighbour does not push this stream"
        );

        let snapshot = pending.snapshot(STREAM);
        pending.retire(STREAM, snapshot);
        assert!(!pending.is_pending(STREAM));
    }

    /// Streams are counted separately: work on one is not work on another.
    #[test]
    fn streams_do_not_share_their_counts() {
        let pending = PendingWork::new();

        pending.mark(OTHER);
        assert!(pending.is_pending(OTHER));
        assert!(!pending.is_pending(STREAM));

        let snapshot = pending.snapshot(STREAM);
        pending.retire(STREAM, snapshot);
        assert!(
            pending.is_pending(OTHER),
            "flushing an idle stream pushes nothing of B"
        );

        let snapshot = pending.snapshot(OTHER);
        pending.retire(OTHER, snapshot);
        assert!(!pending.is_pending(OTHER));
    }
}
