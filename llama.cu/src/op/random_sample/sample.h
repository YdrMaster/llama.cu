#ifndef __SAMPLE_H__
#define __SAMPLE_H__

#include "export.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__C __export cudaError_t calculate_workspace_size_half(
    size_t *argmax,
    size_t *random_sample,
    size_t n);

__C __export cudaError_t argmax_half(
    void *kv_pair,
    void const *logits,
    size_t n,
    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream);

__C __export cudaError_t sample_half(
    void *kv_pair,
    void const *logits,
    unsigned int const *indices,
    size_t n,
    float random,
    float temperature,
    float topp,
    size_t topk,
    void *workspace_ptr,
    size_t workspace_len,
    cudaStream_t stream);

#endif // __SAMPLE_H__
