use anyhow::Result;
use tokio::sync::mpsc;
use tracing::info;

use crate::{BatchJob, WorkerModel};

pub struct Worker {
    pub id: u32,
    pub inbox: mpsc::Receiver<BatchJob>,
    pub model: Box<dyn WorkerModel>,
}

impl Worker {
    pub async fn run(mut self) -> Result<()> {
        info!(worker_id = self.id, "worker started");
        while let Some(job) = self.inbox.recv().await {
            self.model.infer_batch(job)?;
        }
        Ok(())
    }
}
