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

        /// Path to ONNX model file
        #[arg(long, default_value = "models/demo.onnx")]
        model_path: String,
    },
}
