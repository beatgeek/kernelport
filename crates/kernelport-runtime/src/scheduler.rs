use anyhow::Result;
use tokio::sync::mpsc;

use crate::BatchJob;

#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::Sender<BatchJob>,
}

impl SchedulerHandle {
    pub async fn submit(&self, job: BatchJob) -> Result<()> {
        self.tx
            .send(job)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

pub struct Scheduler {
    rx: mpsc::Receiver<BatchJob>,
    worker_txs: Vec<mpsc::Sender<BatchJob>>,
    rr: usize,
}

impl Scheduler {
    pub fn new(rx: mpsc::Receiver<BatchJob>, worker_txs: Vec<mpsc::Sender<BatchJob>>) -> Self {
        Self {
            rx,
            worker_txs,
            rr: 0,
        }
    }

    pub fn handle(tx: mpsc::Sender<BatchJob>) -> SchedulerHandle {
        SchedulerHandle { tx }
    }

    pub async fn run(mut self) -> Result<()> {
        while let Some(job) = self.rx.recv().await {
            let idx = self.rr % self.worker_txs.len();
            self.rr += 1;
            self.worker_txs[idx]
                .send(job)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        Ok(())
    }
}
