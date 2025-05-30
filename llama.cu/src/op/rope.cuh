template <class Tp, class Ta>
static __device__ void padding(
    Ta *__restrict__ y_,
    int const stride_token_y,
    int const stride_head_y,
    Ta const *__restrict__ x_,
    int const stride_token_x,
    int const stride_head_x,
    Tp const *__restrict__ pos_,
    float const *__restrict__ sin_table,
    float const *__restrict__ cos_table) {

    int nh_l = blockDim.y,
         dh_div_2 = blockDim.x,
         it = blockIdx.y,
         ih_h = blockIdx.x,
         ih_l = threadIdx.y,
         ih = ih_h * nh_l + ih_l,
         i = threadIdx.x;

    // 计算 x 和 y 的位置，i * 2 是因为每两个为一组
    auto x = x_ + it * stride_token_x + ih * stride_head_x + i * 2;
    auto y = y_ + it * stride_token_y + ih * stride_head_y + i * 2;

    // 获取位置索引
    auto pos = pos_[it];
    float sin = sin_table[pos * dh_div_2 + i],
          cos = cos_table[pos * dh_div_2 + i],
          a = x[0],
          b = x[1];

    // 应用旋转并写入 y
    y[0] = Ta(a * cos - b * sin);
    y[1] = Ta(a * sin + b * cos);
}
