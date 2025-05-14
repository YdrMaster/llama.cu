fn main() {
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    assert!(
        find_cuda_root().is_some(),
        "cuda not found, check $CUDA_ROOT env var"
    );
    assert!(
        find_nccl_root().is_some(),
        "nccl not found, check $LIBRARY env var"
    );
}
