fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    assert!(
        find_cuda_root().is_some(),
        "cuda not found, check $CUDA_ROOT env var"
    );
    let nccl = Cfg::new("nccl");
    if find_nccl_root().is_some() {
        nccl.define()
    }
}
