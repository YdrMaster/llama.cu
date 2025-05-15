#include "export.h"
#include "sample.cuh"
// #include "sample.h"

__C __export cudaError calculate_workspace_size_half(
    size_t *argmax,
    size_t *random_sample,
    size_t n) {
    {
        return calculate_workspace_size<half>(argmax, random_sample, n);
    }
}

__C __export cudaError argmax_half(
    cub::KeyValuePair<int, half> *kv_pair,
    half const *logits,
    size_t n,

    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream) {
    {
        return arg_max(
            kv_pair,
            logits,
            n,

            workspace_ptr,
            workspace_len,
            stream);
    }
}

__C __export cudaError sample_half(
    cub::KeyValuePair<int, half> *kv_pair,
    half const *logits,
    unsigned int const *indices,
    size_t n,
    float random,
    float temperature,
    float topp,
    size_t topk,
    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream) {
    {
        return random_sample(
            kv_pair,
            logits,
            indices,
            n,

            random,
            temperature,
            topp,
            topk,

            workspace_ptr,
            workspace_len,
            stream);
    }
}
