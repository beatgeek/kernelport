use anyhow::Result;
use bytes::Bytes;
use kernelport_core::{DType, IOName, Shape, Tensor};
use kernelport_proto::kernelport::v1 as pb;
use kernelport_runtime::{InferenceRequest, InferenceResponse};
use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

pub struct GrpcSvc {
    // v0: route everything into a single batcher input channel.
    pub batcher_tx: mpsc::Sender<InferenceRequest>,
}

#[tonic::async_trait]
impl pb::inference_service_server::InferenceService for GrpcSvc {
    async fn predict(
        &self,
        req: Request<pb::PredictRequest>,
    ) -> std::result::Result<Response<pb::PredictResponse>, Status> {
        let req = req.into_inner();

        let mut inputs = Vec::with_capacity(req.inputs.len());
        for t in req.inputs {
            let dtype =
                parse_dtype(&t.dtype).map_err(|e| Status::invalid_argument(e.to_string()))?;
            let shape_usize: Vec<usize> = t
                .shape
                .into_iter()
                .map(|d| usize::try_from(d).unwrap_or(0))
                .collect();
            let tensor =
                Tensor::from_cpu_bytes(dtype, Shape::from_slice(&shape_usize), Bytes::from(t.data));
            inputs.push((IOName(t.name), tensor));
        }

        let (tx, rx) = oneshot::channel();
        let inf_req = InferenceRequest {
            model: req.model,
            version: None,
            inputs,
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(5),
            resp_tx: tx,
        };

        self.batcher_tx
            .send(inf_req)
            .await
            .map_err(|e| Status::unavailable(e.to_string()))?;

        let InferenceResponse { outputs, timings } =
            rx.await.map_err(|_| Status::internal("worker dropped"))?;

        let mut pb_outs = Vec::with_capacity(outputs.len());
        for (name, t) in outputs {
            let data = match t.storage {
                kernelport_core::TensorStorage::CpuBytes(b) => b,
                kernelport_core::TensorStorage::CpuPinned(p) => p.bytes,
                kernelport_core::TensorStorage::CudaDevice(_d) => Bytes::new(), // v0
            };
            pb_outs.push(pb::Tensor {
                name: name.0,
                dtype: format!("{:?}", t.desc.dtype),
                shape: t.desc.shape.0.iter().map(|d| *d as i64).collect(),
                data: data.to_vec(),
            });
        }

        Ok(Response::new(pb::PredictResponse {
            outputs: pb_outs,
            queued_us: timings.queued_us,
            batched_us: timings.batched_us,
            backend_us: timings.backend_us,
        }))
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "U8" => DType::U8,
        _ => anyhow::bail!("unknown dtype: {}", s),
    })
}
