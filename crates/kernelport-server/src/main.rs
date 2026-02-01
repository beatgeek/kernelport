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
        max_batch: 8,
        max_delay: std::time::Duration::from_millis(5),
    };
    let batcher = Batcher::new(batch_policy, batcher_rx, scheduler_handle);

    // Model registry
    let mut reg = registry::ModelRegistry::new();
    match backend.as_str() {
        "onnx" => {
            reg.load_onnx("demo", model_path, device).ok();
        }
        "helion" => {
            reg.load_helion("demo", helion_addr, helion_model, device)
                .ok();
        }
        other => {
            anyhow::bail!("unsupported backend: {other} (expected onnx or helion)");
        }
    }

    let loaded = reg.get("demo");
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
    loaded: Option<std::sync::Arc<registry::LoadedModel>>,
}

impl WorkerModel for DemoWorkerModel {
    fn infer_batch(&mut self, job: BatchJob) -> anyhow::Result<()> {
        let model = match self.loaded.as_ref() {
            Some(model) => model,
            None => {
                tracing::error!("model not loaded (demo)");
                for req in job.requests {
                    let _ = req.resp_tx.send(kernelport_runtime::InferenceResponse {
                        outputs: Vec::new(),
                        timings: kernelport_runtime::Timings {
                            queued_us: 0,
                            batched_us: 0,
                            backend_us: 0,
                        },
                    });
                }
                return Ok(());
            }
        };

        // v0: no real batching; call model once and fan-out same output
        let mut guard = model.model.lock().unwrap();

        let t0 = std::time::Instant::now();
        let outputs = match guard.infer(job.merged_inputs.into_iter().map(|(_, t)| t).collect()) {
            Ok(outputs) => outputs,
            Err(err) => {
                tracing::error!(error = ?err, "model inference failed (demo)");
                for req in job.requests {
                    let _ = req.resp_tx.send(kernelport_runtime::InferenceResponse {
                        outputs: Vec::new(),
                        timings: kernelport_runtime::Timings {
                            queued_us: 0,
                            batched_us: 0,
                            backend_us: 0,
                        },
                    });
                }
                return Ok(());
            }
        };
        let backend_us = t0.elapsed().as_micros() as u64;

        let output_names = model.output_names.clone();
        for req in job.requests {
            let _ = req.resp_tx.send(kernelport_runtime::InferenceResponse {
                outputs: outputs
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, t)| {
                        let name = output_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| kernelport_core::IOName(format!("out{}", i)));
                        (name, t)
                    })
                    .collect(),
                timings: kernelport_runtime::Timings {
                    queued_us: 0,
                    batched_us: 0,
                    backend_us,
                },
            });
        }
        Ok(())
    }
}
