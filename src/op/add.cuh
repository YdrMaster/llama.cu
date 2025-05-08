template <class Tdata>
static __device__ void kernel(
    Tdata *__restrict__ y,
    int const sny,
    int const sdy,
    Tdata const *__restrict__ x,
    int const snx,
    int const sdx,
    Tdata const *__restrict__ b,
    int const snb,
    int const sdb,
    float k) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x,
         iy = blockIdx.y * sny + i * sdy,
         ix = blockIdx.y * snx + i * sdx,
         ib = blockIdx.y * snb + i * sdb;
    y[iy] = (Tdata)(k == 0 ? (float)b[ib] : (k * (float)x[ix] + (float)b[ib]));
}
