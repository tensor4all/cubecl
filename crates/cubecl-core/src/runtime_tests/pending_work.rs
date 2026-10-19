//! What a client reads about its stream's unflushed work: [`Client::has_pending_work`] is true
//! while a call on the stream submitted something no `flush` has pushed, for every client of
//! that stream and for no other, and a flush clears it.
//!
//! The answer is what lets a caller that interleaves its own device work with `CubeCL`'s skip a
//! flush for a queue nothing was submitted to, so the cases below are the ones it relies on:
//! a fresh stream, a submission from another thread, and a neighbour stream that must not be
//! mistaken for this one.

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// A stream nobody has used holds nothing, an allocation submitted on it does, and a flush
/// pushes it.
pub fn a_fresh_stream_lists_nothing_until_it_allocates<R: Runtime>() {
    let client = R::client(&Default::default());
    let stream = StreamId { value: 4_000_001 };

    stream.executes(|| {
        assert!(
            !client.has_pending_work(),
            "a stream no call has used holds nothing"
        );

        let buffer = client.empty(64);
        assert!(
            client.has_pending_work(),
            "the allocation was handed to the device"
        );

        client.flush().expect("flush");
        assert!(
            !client.has_pending_work(),
            "the flush pushed the allocation"
        );

        drop(buffer);
    });
}

/// A create and a launch are each pending work until a flush pushes them.
pub fn a_launch_is_pending_until_it_is_flushed<R: Runtime>() {
    let client = R::client(&Default::default());
    let stream = StreamId { value: 4_000_002 };

    stream.executes(|| {
        let n = 4usize;
        let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let output = client.empty(n * core::mem::size_of::<f32>());
        assert!(
            client.has_pending_work(),
            "the create and the empty were handed to the device"
        );
        // Caught up first, so only the launch can make the stream pending again.
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the flush pushed the buffers");

        add_one::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(&client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
        assert!(
            client.has_pending_work(),
            "the launch was handed to the device"
        );

        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the flush pushed the launch");
    });
}

/// The calls that answer about a clean stream, or wait on it, submit no work: asking them
/// must not report the stream as pending, or a caller that checks after every one pays a
/// flush for an empty queue.
pub fn reading_the_stream_without_submitting_work_does_not_mark_it<R: Runtime>() {
    let client = R::client(&Default::default());
    let stream = StreamId { value: 4_000_003 };

    stream.executes(|| {
        let buffer = client.empty(64);
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the stream is caught up");

        client.check([&buffer]).expect("check");
        assert!(
            !client.has_pending_work(),
            "the claim check looked at work that was already submitted"
        );

        cubecl_environment::future::block_on(client.sync()).expect("sync");
        assert!(
            !client.has_pending_work(),
            "the barrier waits on work that was already submitted"
        );
    });
}

/// A clone used from another thread on the same stream publishes its mark before it returns,
/// so the thread that joins it sees the work as pending until a flush pushes it.
pub fn work_submitted_from_another_thread_is_pending_on_the_stream<R: Runtime>() {
    let client = R::client(&Default::default());
    let stream = StreamId { value: 4_000_004 };

    stream.executes(|| {
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the stream is caught up");

        let clone = client.clone();
        std::thread::spawn(move || {
            stream.executes(|| drop(clone.empty(64)));
        })
        .join()
        .expect("the submitting thread panicked");

        assert!(
            client.has_pending_work(),
            "the other thread's allocation is not pushed yet"
        );

        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the flush pushed it");
    });
}

/// Work on one stream is not work on another, and flushing one stream does not push the
/// other: a caller with a stream per queue reads each independently.
pub fn a_neighbours_stream_is_not_this_streams_pending_work<R: Runtime>() {
    let client = R::client(&Default::default());
    let first = StreamId { value: 4_000_005 };
    let second = StreamId { value: 4_000_006 };

    first.executes(|| {
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the first stream is caught up");
    });

    second.executes(|| drop(client.empty(64)));
    assert!(
        second.executes(|| client.has_pending_work()),
        "the second stream holds the allocation"
    );

    first.executes(|| {
        assert!(
            !client.has_pending_work(),
            "the second stream's work is not the first's"
        );
        client.flush().expect("flush");
        assert!(
            !client.has_pending_work(),
            "flushing the first stream pushes nothing of the second's"
        );
    });

    second.executes(|| {
        assert!(
            client.has_pending_work(),
            "the second stream still holds its allocation"
        );
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the flush pushed it");
    });
}

/// A flush that runs on the device's own thread pushes only what is already ahead of it
/// there. Inside an [`exclusive`](cubecl_runtime::client::Client::exclusive) section the caller
/// runs on that thread, so its flush must not retire the work it just submitted: that work is
/// still pending outside, until a flush from a caller thread pushes it.
pub fn an_inline_flush_does_not_retire_the_work_around_it<R: Runtime>() {
    let client = R::client(&Default::default());
    let stream = StreamId { value: 4_000_007 };

    stream.executes(|| {
        client.flush().expect("flush");
        assert!(!client.has_pending_work(), "the stream is caught up");

        let inside = client.clone();
        client
            .exclusive(move || {
                drop(inside.empty(64));
                assert!(
                    inside.has_pending_work(),
                    "the allocation was handed to the device"
                );

                inside.flush().expect("flush");
                assert!(
                    inside.has_pending_work(),
                    "a flush on the device thread pushes nothing past what is ahead of it"
                );
            })
            .expect("exclusive");

        assert!(
            client.has_pending_work(),
            "the inline flush retired nothing"
        );
        client.flush().expect("flush");
        assert!(
            !client.has_pending_work(),
            "a flush from the caller thread pushes it"
        );
    });
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_pending_work {
    () => {
        mod pending_work {
            use super::*;

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_fresh_stream_lists_nothing_until_it_allocates() {
                cubecl_core::runtime_tests::pending_work::a_fresh_stream_lists_nothing_until_it_allocates::<
                    TestRuntime,
                >();
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_launch_is_pending_until_it_is_flushed() {
                cubecl_core::runtime_tests::pending_work::a_launch_is_pending_until_it_is_flushed::<
                    TestRuntime,
                >();
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn reading_the_stream_without_submitting_work_does_not_mark_it() {
                cubecl_core::runtime_tests::pending_work::reading_the_stream_without_submitting_work_does_not_mark_it::<
                    TestRuntime,
                >();
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn work_submitted_from_another_thread_is_pending_on_the_stream() {
                cubecl_core::runtime_tests::pending_work::work_submitted_from_another_thread_is_pending_on_the_stream::<
                    TestRuntime,
                >();
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_neighbours_stream_is_not_this_streams_pending_work() {
                cubecl_core::runtime_tests::pending_work::a_neighbours_stream_is_not_this_streams_pending_work::<
                    TestRuntime,
                >();
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn an_inline_flush_does_not_retire_the_work_around_it() {
                cubecl_core::runtime_tests::pending_work::an_inline_flush_does_not_retire_the_work_around_it::<
                    TestRuntime,
                >();
            }
        }
    };
}
