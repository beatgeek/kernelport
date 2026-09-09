use std::time::Duration;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use kernelport_core::{
    Backend, BackendCapabilities, BackendModel, DType, Device, IOName, InvalidRequest,
    ModelArtifact, ModelSpec, Shape, Tensor, TensorSpec, TensorStorage,
};
use kernelport_proto::kernelport::v1 as pb;
use kernelport_proto::kernelport::v1::inference_service_client::InferenceServiceClient;
use tonic::transport::{Channel, Endpoint};

pub struct HelionBackend;

impl HelionBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HelionBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HelionModel {
    spec: ModelSpec,
    client: InferenceServiceClient<Channel>,
    model: String,
    timeout: Duration,
}

impl Backend for HelionBackend {
    type Model = HelionModel;

    fn name(&self) -> &'static str {
        "helion"
    }

    fn load(&self, artifact: &ModelArtifact, _device: Device) -> Result<Self::Model> {
        let ModelArtifact::HelionGrpc { addr, model } = artifact else {
            bail!("helion backend expects a HelionGrpc artifact");
        };

        let endpoint = Endpoint::from_shared(addr.clone())
            .context("invalid helion gRPC address")?
            .connect_lazy();
        let client = InferenceServiceClient::new(endpoint);

        // v0: assume softmax-like 2D f16 tensors with x->y names and dynamic dims.
        let spec = ModelSpec {
            inputs: vec![TensorSpec {
                name: IOName("x".to_string()),
                dtype: DType::F16,
                rank: 2,
                dims: vec![None, None],
            }],
            outputs: vec![TensorSpec {
                name: IOName("y".to_string()),
                dtype: DType::F16,
                rank: 2,
                dims: vec![None, None],
            }],
            max_batch: 1,
        };

        Ok(HelionModel {
            spec,
            client,
            model: model.clone(),
            timeout: Duration::from_secs(120),
        })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_dynamic_shapes: true,
            prefers_nchw: false,
            allows_cuda_graphs: false,
        }
    }
}

impl BackendModel for HelionModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn infer(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let named = inputs
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| {
                let name = self
                    .spec
                    .inputs
                    .get(i)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| IOName(format!("input{i}")));
                (name, tensor)
            })
            .collect();
        Ok(self
            .infer_named(named)?
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect())
    }

    fn infer_named(&mut self, inputs: Vec<(IOName, Tensor)>) -> Result<Vec<(IOName, Tensor)>> {
        let pb_inputs = inputs
            .into_iter()
            .map(|(name, tensor)| tensor_to_pb(&name.0, tensor))
            .collect::<Result<Vec<_>>>()?;

        let mut request = tonic::Request::new(pb::InferRequest {
            model: self.model.clone(),
            inputs: pb_inputs,
        });
        request.set_timeout(self.timeout);

        let mut client = self.client.clone();
        let response: tonic::Response<pb::InferResponse> = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(async { client.infer(request).await })
        })
        .map_err(classify_sidecar_status)?;
        let response = response.into_inner();

        let mut outputs = Vec::with_capacity(response.outputs.len());
        for output in response.outputs {
            outputs.push((IOName(output.name.clone()), pb_to_tensor(output)?));
        }

        Ok(outputs)
    }
}

/// Preserve the sidecar's own classification of a failure.
///
/// The sidecar rejects malformed requests with `INVALID_ARGUMENT` and reports
/// its own faults with `INTERNAL`/`UNAVAILABLE`. Wrapping everything as a
/// server fault would hide caller mistakes behind `INTERNAL` at our boundary.
fn classify_sidecar_status(status: tonic::Status) -> anyhow::Error {
    use tonic::Code;
    let detail = format!("helion inference request failed: {status}");
    match status.code() {
        Code::InvalidArgument | Code::OutOfRange | Code::NotFound | Code::FailedPrecondition => {
            InvalidRequest::err(detail)
        }
        _ => anyhow::anyhow!(detail),
    }
}

fn tensor_to_pb(name: &str, tensor: Tensor) -> Result<pb::Tensor> {
    let data = match tensor.storage {
        TensorStorage::CpuBytes(bytes) => bytes,
        TensorStorage::CpuPinned(p) => p.bytes,
        TensorStorage::CudaDevice(_) => bail!("helion backend only supports CPU tensors"),
    };

    Ok(pb::Tensor {
        name: name.to_string(),
        dtype: to_proto_dtype(tensor.desc.dtype) as i32,
        shape: tensor.desc.shape.0.iter().map(|d| *d as i64).collect(),
        data: data.to_vec(),
    })
}

fn pb_to_tensor(tensor: pb::Tensor) -> Result<Tensor> {
    let dtype = parse_dtype(tensor.dtype)?;
    let shape: Vec<usize> = tensor
        .shape
        .into_iter()
        .map(|d| usize::try_from(d).unwrap_or(0))
        .collect();

    Ok(Tensor::from_cpu_bytes(
        dtype,
        Shape::from_slice(&shape),
        Bytes::from(tensor.data),
    ))
}

fn parse_dtype(raw: i32) -> Result<DType> {
    let dtype = pb::DType::try_from(raw).context("unknown dtype enum value")?;
    Ok(match dtype {
        pb::DType::F32 => DType::F32,
        pb::DType::F16 => DType::F16,
        pb::DType::I64 => DType::I64,
        pb::DType::I32 => DType::I32,
        pb::DType::U8 => DType::U8,
        pb::DType::DtypeUnspecified => bail!("dtype is unspecified"),
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
