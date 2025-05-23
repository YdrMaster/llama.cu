template <class T>
struct KV {
    unsigned int k;
    T v;
};

template <class Tval, class Tidx>
static __device__ void kernel(
    Tidx *__restrict__ tokens,
    KV<Tval> const *__restrict__ kv_pairs,
    KV<Tidx> const *__restrict__ map) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto kv = map[i];
    tokens[kv.v] = kv_pairs[kv.k].k;
}
