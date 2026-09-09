mod cli;
mod grpc;
mod registry;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use kernelport_proto::kernelport::v1::inference_service_server::InferenceServiceServer;
use kernelport_runtime::{BatchPolicy, Batcher, Scheduler, Worker};
use tokio::sync::mpsc;
use tonic_reflection::server::Builder as ReflectionBuilder;
use tracing_subscriber::EnvFilter;

use grpc::GrpcSvc;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve {
            grpc_addr,
            log,
            device,
            backend,
            model_path,
            helion_addr,
            helion_model,
        } => {
            let device = parse_device(&device)?;
            serve(
                grpc_addr,
                log,
                device,
                backend,
                model_path.into(),
                helion_addr,
                helion_model,
            )
            .await
        }
    }
}

async fn serve(
    grpc_addr: String,
    log: String,
    device: kernelport_core::Device,
    backend: String,
    model_path: std::path::PathBuf,
    helion_addr: String,
    helion_model: String,
) -> Result<()> {
    std::env::set_var("RUST_LOG", &log);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // ---- Channels: requests -> batcher -> scheduler -> worker(s)
    let (batcher_tx, batcher_rx) = mpsc::channel(1024);
    let (sched_tx, sched_rx) = mpsc::channel(1024);

    // v0: one worker
    let (w_tx, w_rx) = mpsc::channel(128);

    let scheduler = Scheduler::new(sched_rx, vec![w_tx]);
    let scheduler_handle = Scheduler::handle(sched_tx);

    let batch_policy = BatchPolicy {
        max_batch: 1,
        max_delay: std::time::Duration::ZERO,
    };
    let batcher = Batcher::new(batch_policy, batcher_rx, scheduler_handle);

    // Model registry
    let mut reg = registry::ModelRegistry::new();
    match backend.as_str() {
        "onnx" => {
            reg.load_onnx("demo", model_path, device)?;
        }
        "helion" => {
            reg.load_helion("demo", helion_addr, helion_model, device)?;
        }
        other => {
            anyhow::bail!("unsupported backend: {other} (expected onnx or helion)");
        }
    }

    let loaded = reg.get("demo").context("model not loaded")?;
    let worker_model = DemoWorkerModel { loaded };

    let worker = Worker {
        id: 0,
        inbox: w_rx,
        model: Box::new(worker_model),
    };

    // Run components
    tokio::spawn(async move {
        if let Err(e) = scheduler.run().await {
            tracing::error!(error=?e, "scheduler exited");
        }
    });
    tokio::spawn(async move {
        if let Err(e) = batcher.run().await {
            tracing::error!(error=?e, "batcher exited");
        }
    });
    tokio::spawn(async move {
        if let Err(e) = worker.run().await {
            tracing::error!(error=?e, "worker exited");
        }
    });

    // gRPC server
    let addr = grpc_addr.parse()?;
    let svc = GrpcSvc { batcher_tx };

    tracing::info!(%addr, "kernelportd gRPC listening");
    let reflection = ReflectionBuilder::configure()
        .register_encoded_file_descriptor_set(kernelport_proto::FILE_DESCRIPTOR_SET)
        .build_v1()
        .map_err(|e| anyhow::anyhow!("reflection build failed: {e}"))?;

    tonic::transport::Server::builder()
        .add_service(InferenceServiceServer::new(svc))
        .add_service(reflection)
        .serve(addr)
        .await?;

    Ok(())
}

fn parse_device(raw: &str) -> Result<kernelport_core::Device> {
    if raw.eq_ignore_ascii_case("cpu") {
        return Ok(kernelport_core::Device::Cpu);
    }

    if let Some(rest) = raw.strip_prefix("cuda:") {
        let device_id: u32 = rest.parse().context("invalid cuda device id")?;
        return Ok(kernelport_core::Device::Cuda { device_id });
    }

    anyhow::bail!("unsupported device: {raw} (expected cpu or cuda:N)");
}

use kernelport_runtime::{BatchJob, WorkerModel};

struct DemoWorkerModel {
    loaded: std::sync::Arc<registry::LoadedModel>,
}

impl WorkerModel for DemoWorkerModel {
    fn infer_batch(&mut self, job: BatchJob) -> anyhow::Result<()> {
        // Until true stacking/splitting exists, execute each request independently.
        // Never fan out one request's result to other callers.
        let mut guard = self
            .loaded
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        for req in job.requests {
            if req.resp_tx.is_closed() {
                continue;
            }
            let t0 = std::time::Instant::now();
            let result = guard
                .infer(req.inputs)
                .map(|outputs| kernelport_runtime::InferenceResponse {
                    outputs,
                    timings: kernelport_runtime::Timings {
                        queued_us: t0.duration_since(job.created_at).as_micros() as u64,
                        batched_us: 0,
                        backend_us: t0.elapsed().as_micros() as u64,
                    },
                })
                .map_err(|err| kernelport_runtime::InferError::from_backend(&err));
            let _ = req.resp_tx.send(result);
        }
        Ok(())
    }
}
