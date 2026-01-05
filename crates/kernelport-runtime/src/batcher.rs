use anyhow::Result;
use kernelport_core::{IOName, Tensor};
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};
use tracing::debug;

use crate::{InferenceRequest, SchedulerHandle};

#[derive(Clone, Debug)]
pub struct BatchPolicy {
    pub max_batch: usize,
    pub max_delay: Duration,
}

/// A batch ready to run on a worker.
#[derive(Debug)]
pub struct BatchJob {
    pub model: String,
    pub requests: Vec<InferenceRequest>,
    pub merged_inputs: Vec<(IOName, Tensor)>,
    pub created_at: std::time::Instant,
}

pub struct Batcher {
    policy: BatchPolicy,
    rx: mpsc::Receiver<InferenceRequest>,
    scheduler: SchedulerHandle,
}

impl Batcher {
    pub fn new(
        policy: BatchPolicy,
        rx: mpsc::Receiver<InferenceRequest>,
        scheduler: SchedulerHandle,
    ) -> Self {
        Self {
            policy,
            rx,
            scheduler,
        }
    }

    pub async fn run(mut self) -> Result<()> {
        let mut pending: Vec<InferenceRequest> = Vec::new();
        let mut first_seen: Option<Instant> = None;

        loop {
            tokio::select! {
                maybe_req = self.rx.recv() => {
                    match maybe_req {
                        None => break,
                        Some(req) => {
                            if pending.is_empty() { first_seen = Some(Instant::now()); }
                            pending.push(req);
                            if pending.len() >= self.policy.max_batch {
                                self.flush(&mut pending).await?;
                                first_seen = None;
                            }
                        }
                    }
                }
                _ = async {
                    // When we have at least one pending request, sleep until the batching window expires.
                    if let Some(t0) = first_seen {
                        sleep(self.policy.max_delay.saturating_sub(t0.elapsed())).await;
                    } else {
                        sleep(Duration::from_millis(1)).await;
                    }
                }, if first_seen.is_some() => {
                    if !pending.is_empty() {
                        self.flush(&mut pending).await?;
                        first_seen = None;
                    }
                }
            }
        }

        Ok(())
    }

    async fn flush(&self, pending: &mut Vec<InferenceRequest>) -> Result<()> {
        let reqs = std::mem::take(pending);

        // Defensive: should never happen, but avoids panic if caller changes logic.
        if reqs.is_empty() {
            return Ok(());
        }

        let model = reqs[0].model.clone();

        // v0: require identical inputs & shapes; "pretend merged" by using first request's inputs.
        // Next step: stack inputs along batch dimension.
        let merged_inputs = reqs[0].inputs.clone();

        debug!(model=%model, batch=reqs.len(), "dispatching batch");
        let job = BatchJob {
            model,
            requests: reqs,
            merged_inputs,
            created_at: std::time::Instant::now(),
        };

        self.scheduler.submit(job).await?;
        Ok(())
    }
}
