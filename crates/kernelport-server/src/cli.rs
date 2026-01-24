use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "kernelportd", version, about = "KernelPort inference daemon")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the inference server
    Serve {
        /// Bind address for gRPC
        #[arg(long, default_value = "0.0.0.0:50051")]
        grpc_addr: String,

        /// Log level (RUST_LOG)
        #[arg(long, default_value = "info")]
        log: String,

        /// Device for inference (cpu or cuda:N)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Backend type (onnx or helion)
        #[arg(long, default_value = "onnx")]
        backend: String,

        /// Path to ONNX model file
        #[arg(long, default_value = "models/demo.onnx")]
        model_path: String,

        /// Helion sidecar address (gRPC)
        #[arg(long, default_value = "http://127.0.0.1:50061")]
        helion_addr: String,

        /// Helion kernel entrypoint name
        #[arg(long, default_value = "softmax_two_pass")]
        helion_model: String,
    },
}
