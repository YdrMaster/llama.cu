static __forceinline__ __device__ float sigmoid(float x) {
    return fdividef(1, 1 + expf(-x));
}

template <class Tdata>
static __device__ void kernel(
    Tdata *__restrict__ out,
    int const stride_out,
    Tdata const *__restrict__ gate_,
    int const stride_gate,
    Tdata const *__restrict__ up_,
    int const stride_up) {
    auto n = blockIdx.x * blockDim.x + threadIdx.x,
         i = blockIdx.y * stride_out + n,
         j = blockIdx.y * stride_gate + n,
         k = blockIdx.y * stride_up + n;
    float gate = gate_[j],
          up = up_[k];
    out[i] = Tdata(gate * sigmoid(gate) * up);
}
