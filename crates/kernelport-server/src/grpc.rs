use anyhow::{Context, Result};
use bytes::Bytes;
use kernelport_core::{DType, IOName, Shape, Tensor};
use kernelport_proto::kernelport::v1 as pb;
use kernelport_runtime::{InferError, InferenceRequest, InferenceResponse};
use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

pub struct GrpcSvc {
    // v0: route everything into a single batcher input channel.
    pub batcher_tx: mpsc::Sender<InferenceRequest>,
}

#[tonic::async_trait]
impl pb::inference_service_server::InferenceService for GrpcSvc {
    async fn infer(
        &self,
        req: Request<pb::InferRequest>,
    ) -> std::result::Result<Response<pb::InferResponse>, Status> {
        let req = req.into_inner();

        if req.model != "demo" {
            return Err(Status::not_found("unknown model"));
        }
        if req.inputs.is_empty() {
            return Err(Status::invalid_argument("no inputs provided"));
        }
        let mut inputs = Vec::with_capacity(req.inputs.len());
        for t in req.inputs {
            let dtype =
                parse_dtype(t.dtype).map_err(|e| Status::invalid_argument(e.to_string()))?;
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

        let InferenceResponse { outputs, timings } = rx
            .await
            .map_err(|_| Status::internal("worker dropped"))?
            .map_err(|err| match err {
                // Caller-caused failures must stay distinguishable from server
                // faults; clients and tests key retry behaviour off the code.
                InferError::InvalidArgument(message) => Status::invalid_argument(message),
                InferError::Internal(message) => Status::internal(message),
            })?;

        let mut pb_outs = Vec::with_capacity(outputs.len());
        for (name, t) in outputs {
            let data = match t.storage {
                kernelport_core::TensorStorage::CpuBytes(b) => b,
                kernelport_core::TensorStorage::CpuPinned(p) => p.bytes,
                kernelport_core::TensorStorage::CudaDevice(_d) => Bytes::new(), // v0
            };
            pb_outs.push(pb::Tensor {
                name: name.0,
                dtype: to_proto_dtype(t.desc.dtype) as i32,
                shape: t.desc.shape.0.iter().map(|d| *d as i64).collect(),
                data: data.to_vec(),
            });
        }

        Ok(Response::new(pb::InferResponse {
            outputs: pb_outs,
            queued_us: timings.queued_us,
            batched_us: timings.batched_us,
            backend_us: timings.backend_us,
        }))
    }
}

fn parse_dtype(raw: i32) -> Result<DType> {
    let dtype = pb::DType::try_from(raw).context("unknown dtype enum value")?;
    Ok(match dtype {
        pb::DType::F32 => DType::F32,
        pb::DType::F16 => DType::F16,
        pb::DType::I64 => DType::I64,
        pb::DType::I32 => DType::I32,
        pb::DType::U8 => DType::U8,
        pb::DType::DtypeUnspecified => anyhow::bail!("dtype is unspecified"),
    })
}

fn to_proto_dtype(dtype: DType) -> pb::DType {
    match dtype {
        DType::F32 => pb::DType::F32,
        DType::F16 => pb::DType::F16,
        DType::I64 => pb::DType::I64,
        DType::I32 => pb::DType::I32,
        DType::U8 => pb::DType::U8,
    }
}
