pub mod kernelport {
    pub mod v1 {
        tonic::include_proto!("kernelport.v1");
    }
}

pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("kernelport_descriptor");
