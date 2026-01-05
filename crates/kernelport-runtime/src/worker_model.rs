use anyhow::Result;

use crate::BatchJob;

/// Object-safe worker model interface.
/// Keep it synchronous; the worker task can call it directly.
pub trait WorkerModel: Send {
    fn infer_batch(&mut self, job: BatchJob) -> Result<()>;
}

