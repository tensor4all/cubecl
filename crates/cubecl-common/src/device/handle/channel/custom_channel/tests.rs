use super::*;
use crate::device::{DeviceId, DeviceServiceStage};
use std::sync::{atomic::AtomicBool, mpsc};

fn runner_id() -> RunnerId {
    RunnerId {
        device: DeviceId {
            type_id: 999,
            index_id: 1,
        },
        stage: DeviceServiceStage::Downstream,
    }
}

#[test]
fn client_registers_wake_target_before_server_initialization_finishes() {
    let (release, wait) = mpsc::channel();
    let client = DeviceClient::new(runner_id(), move || {
        wait.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    assert!(client.state.thread.get().is_some());
    let (tx, rx) = mpsc::channel();
    client.enqueue(move || tx.send(()).unwrap()).unwrap();
    client.flush();
    release.send(()).unwrap();
    rx.recv_timeout(Duration::from_secs(5)).unwrap();
}

#[test]
fn concurrent_publishers_wake_and_execute_each_task_once() {
    let client = DeviceClient::new(runner_id(), || {});
    std::thread::sleep(Duration::from_millis(10));
    let (tx, rx) = mpsc::channel();
    let mut workers = Vec::new();
    for producer in 0..4 {
        let client = client.clone();
        let tx = tx.clone();
        workers.push(std::thread::spawn(move || {
            for index in 0..64 {
                let tx = tx.clone();
                client
                    .enqueue(move || tx.send(producer * 64 + index).unwrap())
                    .unwrap();
                if index % 7 == 0 {
                    client.flush();
                }
            }
            client.flush();
        }));
    }
    let mut seen = [false; 256];
    for _ in 0..seen.len() {
        let index = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!seen[index], "task executed twice");
        seen[index] = true;
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(seen.into_iter().all(|value| value));
}

#[test]
fn full_and_partial_batches_wake_after_idle() {
    let client = DeviceClient::new(runner_id(), || {});
    for size in [CHANNEL_MAX_TASK, 1, CHANNEL_MAX_TASK, 1] {
        std::thread::sleep(Duration::from_millis(10));
        let (tx, rx) = mpsc::channel();
        for index in 0..size {
            let tx = tx.clone();
            client.enqueue(move || tx.send(index).unwrap()).unwrap();
        }
        if size < CHANNEL_MAX_TASK {
            client.flush();
        }
        for index in 0..size {
            assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), index);
        }
    }
}

#[test]
fn ready_batch_before_server_registration_is_processed() {
    let mut server = Server::new(runner_id());
    let client = DeviceClient {
        state: server.state.clone(),
    };
    let (tx, rx) = mpsc::channel();
    client.enqueue(move || tx.send(()).unwrap()).unwrap();
    client.flush();
    assert!(client.state.thread.get().is_none());
    std::thread::spawn(move || server.start());
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
}

#[test]
fn notification_between_queue_check_and_park_is_retained() {
    let state = Server::new(runner_id()).state;
    let proceed = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let worker_state = state.clone();
    let worker_proceed = proceed.clone();
    let worker = std::thread::spawn(move || {
        worker_state.thread.get_or_init(std::thread::current);
        assert_eq!(worker_state.enqueued_count.load(Ordering::Acquire), 0);
        tx.send(()).unwrap();
        while !worker_proceed.load(Ordering::Acquire) {
            spin_loop();
        }
        std::thread::park();
        tx.send(()).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    state.publish_tasks(CHANNEL_MAX_TASK as u32);
    proceed.store(true, Ordering::Release);
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    worker.join().unwrap();
}
