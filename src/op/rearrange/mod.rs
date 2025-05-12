use crate::ArrayLayout;
use itertools::Itertools;
use mem_rearrange::Rearranging;
use operators::cuda::{DevByte, Device, Dim3, Module, Ptx, Stream, params};
use std::cmp::max;
use std::collections::HashMap;
use std::ffi::CString;

use super::ModuleKey;
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SchemeKey {
    unit_size: usize,
    block_array_size: usize,
    grid_array_size: usize,
    constrain_num: usize,
}

impl SchemeKey {
    fn to_module_keys(&self) -> Box<[ModuleKey]> {
        Box::new([
            ModuleKey::Size(self.unit_size),
            ModuleKey::Size(self.block_array_size),
            ModuleKey::Size(self.grid_array_size),
            ModuleKey::Size(self.constrain_num),
        ])
    }
}

/// Type used for array indices and strides
type ArrayType = i32;

// 默认的数组大小，同时也是最大的数组大小，不能为0
const DEFAULT_ARRAY_SIZE: usize = 5;
const CONSTRAIN_ARRAY_SIZE: usize = 8;

#[derive(Debug)]
struct SplitDim {
    choose_idx: usize,
    num_per_block: usize,
    num_per_grid: usize,
    array_struct_idx_block: ArrayType,
    array_struct_idx_grid: ArrayType,
    dim_len: usize,
}

#[derive(Debug)]
struct ArrayStruct(Vec<ArrayType>);

#[derive(Debug)]
enum SchemeError {
    RankNotSupport,
    ShapeMismatch,
    DimReduce,
}

impl From<mem_rearrange::SchemeError> for SchemeError {
    fn from(err: mem_rearrange::SchemeError) -> Self {
        match err {
            mem_rearrange::SchemeError::ShapeMismatch => SchemeError::ShapeMismatch,
            mem_rearrange::SchemeError::DimReduce => SchemeError::DimReduce,
        }
    }
}

impl ArrayStruct {
    fn new(mut array: Vec<ArrayType>, default: ArrayType) -> Self {
        while array.len() < DEFAULT_ARRAY_SIZE {
            array.push(default);
        }
        Self(array)
    }

    fn try_into_array<const N: usize>(self) -> Result<[ArrayType; N], SchemeError> {
        let ArrayStruct(vec) = self;
        if vec.len() <= N {
            Ok(std::array::from_fn(|i| vec.get(i).copied().unwrap_or(0)))
        } else {
            Err(SchemeError::RankNotSupport)
        }
    }
}

pub struct Rearrange {
    #[allow(unused)]
    max_warps_block: usize,
    #[allow(unused)]
    warp_size: usize,
}

enum ShemedRearrange {
    Copied {
        len: usize,
        dst_offset: isize,
        src_offset: isize,
    },
    NotCopied {
        params_without_ptr: ParamsWithoutPtr,
        scheme_key: SchemeKey,
        attrs: (Dim3, Dim3, usize),
        dst_offset: isize,
        src_offset: isize,
    },
}

struct ParamsWithoutPtr {
    block_dim: i32,
    block_len_total: u32,
    block_len: [ArrayType; DEFAULT_ARRAY_SIZE], // 各维度的长度
    src_block_stride: [ArrayType; DEFAULT_ARRAY_SIZE], // 源tensor在各维度上的步长(bytes)
    dst_block_stride: [ArrayType; DEFAULT_ARRAY_SIZE], // 目标tensor在各维度上的步长(bytes)
    grid_len: [ArrayType; DEFAULT_ARRAY_SIZE],  // 各维度的长度
    src_grid_stride: [ArrayType; DEFAULT_ARRAY_SIZE], // 源tensor在各维度上的步长(bytes)
    dst_grid_stride: [ArrayType; DEFAULT_ARRAY_SIZE], // 目标tensor在各维度上的步长(bytes)
    constrains: [ArrayType; CONSTRAIN_ARRAY_SIZE],
}

const CODE: &str = include_str!("rearrange.cuh");

impl Rearrange {
    fn new(dev: &Device) -> Self {
        // 提取和检查设备参数
        let max_threads_block = dev.block_limit().max_threads;
        let warp_size = dev.warp_size();
        assert_eq!(max_threads_block % warp_size, 0);
        // 生成执行资源
        Self {
            max_warps_block: max_threads_block / warp_size,
            warp_size,
        }
    }

    fn scheme<const N: usize>(
        &self,
        dev: &Device,
        dst: &ArrayLayout<N>,
        src: &ArrayLayout<N>,
        unit: usize,
    ) -> Result<ShemedRearrange, SchemeError> {
        let scheme_update = Rearranging::new(dst, src, unit)?;

        // 发现最大的1 thread 处理的数据量
        let scheme_update = scheme_update.distribute_unit((0..=5).rev().map(|n| (1 << n)));
        if scheme_update.ndim() == 0 {
            let unit = scheme_update.unit();
            return Ok(ShemedRearrange::Copied {
                len: unit,
                dst_offset: dst.offset(),
                src_offset: src.offset(),
            });
        }

        let src_strides = scheme_update.src_strides();
        let dst_strides = scheme_update.dst_strides();
        let shape = scheme_update.shape().collect::<Vec<_>>();
        let unit = scheme_update.unit();
        let ndim = scheme_update.ndim();

        //src strides 降序 index
        let src_strides_desc_idx = (0..scheme_update.ndim())
            .zip(src_strides)
            .sorted_by(|a, b| b.1.cmp(a.1))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        //分离维度，分成grid处理的维度和block处理的维度，与dst的维度相对应
        let mut block_dim_choose = vec![false; ndim];

        //TODO 可能需要优化
        let max_block_size = dev.block_limit().max_threads;
        let mut split_dims = Vec::new(); // 长度最多为2

        //进行维度选择
        {
            let mut src_choose_idx = ndim;
            let mut dst_choose_idx = ndim;

            let mut block_elements = 1;
            let mut block_src_elements = 1;
            let mut block_dst_elements = 1;

            while src_choose_idx > 0 && dst_choose_idx > 0 {
                let src_idx = src_strides_desc_idx[src_choose_idx - 1];
                let dst_idx = dst_choose_idx - 1;

                if src_idx == dst_idx {
                    let idx = src_idx;
                    let len = shape[idx];
                    if block_elements * shape[src_idx] <= max_block_size {
                        //选择维度
                        block_dim_choose[idx] = true;
                        block_elements *= len;
                        block_src_elements *= len;
                        block_dst_elements *= len;
                        src_choose_idx -= 1;
                        dst_choose_idx -= 1;
                    } else {
                        //切分维度，并退出
                        let num_per_block = max_block_size.div_euclid(block_elements);
                        assert!(num_per_block > 0);
                        assert!(len >= num_per_block);
                        if num_per_block > 1 {
                            split_dims.push(SplitDim {
                                choose_idx: idx,
                                num_per_block,
                                num_per_grid: len.div_ceil(num_per_block),
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: len,
                            });
                        }
                        break;
                    }
                } else {
                    let src_div_dst = block_src_elements as f64 / block_dst_elements as f64;
                    let src_num_per_block =
                        (max_block_size as f64 / block_elements as f64 / src_div_dst).sqrt();
                    let dst_num_per_block = src_num_per_block * src_div_dst;

                    let src_current_dim_len = shape[src_idx];
                    let dst_current_dim_len = shape[dst_idx];

                    if (src_current_dim_len as f64) < src_num_per_block {
                        //选择维度
                        block_dim_choose[src_idx] = true;
                        block_elements *= src_current_dim_len;
                        block_src_elements *= src_current_dim_len;
                        src_choose_idx -= 1;
                    } else if (dst_current_dim_len as f64) < dst_num_per_block {
                        //选择维度
                        block_dim_choose[dst_idx] = true;
                        block_elements *= dst_current_dim_len;
                        block_dst_elements *= dst_current_dim_len;
                        dst_choose_idx -= 1;
                    } else {
                        //切分维度，并退出
                        let src_num_per_block = src_num_per_block.floor() as usize;
                        let dst_num_per_block = dst_num_per_block.floor() as usize;
                        let src_num_per_grid = src_current_dim_len.div_ceil(src_num_per_block);
                        let dst_num_per_grid = dst_current_dim_len.div_ceil(dst_num_per_block);

                        if src_num_per_block == 1 {
                        } else if src_num_per_grid == 1 {
                            block_dim_choose[src_idx] = true;
                        } else {
                            split_dims.push(SplitDim {
                                choose_idx: src_idx,
                                num_per_block: src_num_per_block,
                                num_per_grid: src_num_per_grid,
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: src_current_dim_len,
                            });
                        }

                        if dst_num_per_block == 1 {
                        } else if dst_num_per_grid == 1 {
                            block_dim_choose[dst_idx] = true;
                        } else {
                            split_dims.push(SplitDim {
                                choose_idx: dst_idx,
                                num_per_block: dst_num_per_block,
                                num_per_grid: dst_num_per_grid,
                                array_struct_idx_block: 0,
                                array_struct_idx_grid: 0,
                                dim_len: dst_current_dim_len,
                            });
                        }
                        break;
                    }
                }
            }
        }

        let mut block_dim: ArrayType = 0;

        let mut block_len = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut src_block_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut dst_block_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);

        let mut grid_len = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut src_grid_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);
        let mut dst_grid_stride = Vec::<ArrayType>::with_capacity(DEFAULT_ARRAY_SIZE);

        // 处理block，填充block_len，block_stride
        for i in 0..ndim {
            if block_dim_choose[i] {
                block_len.push(shape[i] as ArrayType);
                src_block_stride.push(src_strides[i] as ArrayType);
                dst_block_stride.push(dst_strides[i] as ArrayType);
                block_dim += 1;
            }

            for split_dim in split_dims.iter_mut() {
                if i == split_dim.choose_idx {
                    block_len.push(split_dim.num_per_block as ArrayType);
                    src_block_stride.push(src_strides[i] as ArrayType);
                    dst_block_stride.push(dst_strides[i] as ArrayType);
                    split_dim.array_struct_idx_block = block_dim;
                    block_dim += 1;
                }
            }
        }

        // 处理grid，填充grid_len，grid_stride
        let mut grid_dim = 0_u32;
        for i in 0..ndim {
            let mut is_split = false;
            if !block_dim_choose[i] {
                for split_dim in split_dims.iter_mut() {
                    if i == split_dim.choose_idx {
                        is_split = true;
                        grid_len.push(split_dim.num_per_grid as ArrayType);
                        src_grid_stride
                            .push((src_strides[i] * split_dim.num_per_block as isize) as ArrayType);
                        dst_grid_stride
                            .push((dst_strides[i] * split_dim.num_per_block as isize) as ArrayType);
                        split_dim.array_struct_idx_grid = grid_dim as ArrayType;
                    }
                }
                if !is_split {
                    grid_len.push(shape[i] as ArrayType);
                    src_grid_stride.push(src_strides[i] as ArrayType);
                    dst_grid_stride.push(dst_strides[i] as ArrayType);
                }
                grid_dim += 1;
            }
        }

        let filter_split_dims = split_dims
            .iter()
            .filter(|split_dim| split_dim.dim_len % split_dim.num_per_block != 0)
            .collect::<Vec<_>>();

        let constrain_num = filter_split_dims.len();

        // 准备kernel
        let key = SchemeKey {
            unit_size: unit,
            constrain_num,
            block_array_size: block_len.len(),
            grid_array_size: grid_len.len(),
        };

        // 计算grid和block
        let grid = grid_len.iter().product::<ArrayType>() as u32;
        let block = block_len.iter().product::<ArrayType>() as u32;

        // cuda 参数准备
        let block_len_total = block_len.iter().map(|x| *x as u32).product::<u32>();
        let src_block_stride = ArrayStruct::new(src_block_stride, 0);
        let dst_block_stride = ArrayStruct::new(dst_block_stride, 0);
        let src_grid_stride = ArrayStruct::new(src_grid_stride, 0);
        let dst_grid_stride = ArrayStruct::new(dst_grid_stride, 0);
        let block_len = ArrayStruct::new(block_len, 1);
        let grid_len = ArrayStruct::new(grid_len, 1);

        let constrains = match filter_split_dims.len() {
            0 => ArrayStruct(vec![0; 8]),
            1 => ArrayStruct(vec![
                filter_split_dims[0].array_struct_idx_grid,
                filter_split_dims[0].array_struct_idx_block,
                filter_split_dims[0].num_per_block as ArrayType,
                filter_split_dims[0].dim_len as ArrayType,
                0,
                0,
                0,
                0,
            ]),
            2 => ArrayStruct(vec![
                filter_split_dims[0].array_struct_idx_grid,
                filter_split_dims[0].array_struct_idx_block,
                filter_split_dims[0].num_per_block as ArrayType,
                filter_split_dims[0].dim_len as ArrayType,
                filter_split_dims[1].array_struct_idx_grid,
                filter_split_dims[1].array_struct_idx_block,
                filter_split_dims[1].num_per_block as ArrayType,
                filter_split_dims[1].dim_len as ArrayType,
            ]),
            _ => unreachable!(),
        };

        let params_without_ptr = ParamsWithoutPtr {
            block_dim,
            block_len_total,
            block_len: block_len.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 各维度的长度
            src_block_stride: src_block_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 源tensor在各维度上的步长(bytes)
            dst_block_stride: dst_block_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 目标tensor在各维度上的步长(bytes)
            grid_len: grid_len.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 各维度的长度
            src_grid_stride: src_grid_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 源tensor在各维度上的步长(bytes)
            dst_grid_stride: dst_grid_stride.try_into_array::<DEFAULT_ARRAY_SIZE>()?, // 目标tensor在各维度上的步长(bytes)
            constrains: constrains.try_into_array::<CONSTRAIN_ARRAY_SIZE>()?,
        };

        Ok(ShemedRearrange::NotCopied {
            params_without_ptr,
            scheme_key: key,
            attrs: (grid.into(), block.into(), unit),
            dst_offset: dst.offset(),
            src_offset: src.offset(),
        })
    }
}

impl ShemedRearrange {
    pub fn launch<'ctx>(
        &self,
        dst: *mut DevByte,
        src: *const DevByte,
        stream: &Stream<'ctx>,
        modules: &mut HashMap<Box<[ModuleKey]>, Module<'ctx>>,
    ) {
        let ctx = stream.ctx();
        match self {
            ShemedRearrange::Copied {
                len,
                dst_offset,
                src_offset,
            } => {
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(dst.byte_offset(*dst_offset), *len) };
                let src = unsafe { std::slice::from_raw_parts(src.byte_offset(*src_offset), *len) };
                stream.memcpy_d2d(dst, src);
            }
            ShemedRearrange::NotCopied {
                params_without_ptr,
                scheme_key,
                attrs,
                dst_offset,
                src_offset,
            } => {
                let params = params![
                    unsafe { dst.byte_offset(*dst_offset) },
                    unsafe { src.byte_offset(*src_offset) },
                    params_without_ptr.block_dim,
                    params_without_ptr.block_len_total,
                    params_without_ptr.block_len,
                    params_without_ptr.src_block_stride,
                    params_without_ptr.dst_block_stride,
                    params_without_ptr.grid_len,
                    params_without_ptr.src_grid_stride,
                    params_without_ptr.dst_grid_stride,
                    params_without_ptr.constrains
                ];
                let module = modules
                    .entry(scheme_key.to_module_keys())
                    .or_insert_with(|| {
                        let (ptx, log) =
                            Ptx::compile(format_code(*scheme_key), ctx.dev().compute_capability());
                        let Ok(ptx) = ptx else { panic!("{log}") };
                        ctx.load(&ptx)
                    });

                let kernel = module.get_kernel(CString::new(kernel_name(scheme_key)).unwrap());
                stream.launch(&kernel, attrs.clone(), &params.to_ptrs());
            }
        }
    }
}

fn kernel_name(
    SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    }: &SchemeKey,
) -> String {
    let tmem_type = match unit_size {
        1 => "uchar1",
        2 => "uchar2",
        4 => "float1",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => unreachable!(),
    };
    format!(
        "rearrange_unit_{tmem_type}_block_{block_array_size}_grid_{grid_array_size}_constrain_{constrain_num}"
    )
}

fn format_code(
    SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    }: SchemeKey,
) -> String {
    assert!(block_array_size != 0);

    let kernel_name = kernel_name(&SchemeKey {
        unit_size,
        block_array_size,
        grid_array_size,
        constrain_num,
    });
    //处理 grid_array_size = 0的情况
    let grid_array_size = max(grid_array_size, 1);

    let mut code = String::new();

    let tmem_type = match unit_size {
        1 => "uchar1",
        2 => "uchar2",
        4 => "float1",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => unreachable!(),
    };

    // 添加头部定义
    code.push_str(&format!("#define BLOCK_ARRAY_SIZE {block_array_size}\n"));
    code.push_str(&format!("#define GRID_ARRAY_SIZE {grid_array_size}\n"));
    code.push_str("#define ARRAY_TYPE int\n");
    code.push_str(&format!("#define CONSTRAIN_NUM {constrain_num}\n"));
    code.push_str(CODE);
    code.push('\n');

    // 添加实例化宏调用
    code.push_str(&format!(
        r#"
extern "C" __global__ void {kernel_name}(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> block_len,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> src_block_stride,
    const ArrayStruct<BLOCK_ARRAY_SIZE, ARRAY_TYPE> dst_block_stride,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> grid_len,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> src_grid_stride,
    const ArrayStruct<GRID_ARRAY_SIZE, ARRAY_TYPE> dst_grid_stride
#if CONSTRAIN_NUM > 0
    ,const ArrayStruct<CONSTRAIN_NUM, Constrains<ARRAY_TYPE>> constrains
#endif
) {{
    rearrange_kernel<{tmem_type}, {constrain_num}>(
        dst, src, block_dim, block_len_total,
        block_len, src_block_stride, dst_block_stride,
        grid_len, src_grid_stride, dst_grid_stride
#if CONSTRAIN_NUM > 0
        ,constrains
#endif
    );
}}
"#
    ));
    code.push('\n');

    code
}

#[cfg(test)]
mod test {
    use super::SchemeKey;
    use super::{format_code, kernel_name};
    use operators::cuda::{self, Device, Gpu, Ptx};
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::time::Duration;

    #[test]
    fn test_compile() {
        assert!(cuda::init().is_ok());
        let dev = Device::new(0);
        let gpu = Gpu::new(dev.context(), Default::default());

        let mut ptxs = HashMap::new();

        // 遍历所有可能的unit_size和constrain_num组合，编译所有kernel
        for unit_size in (0..=5).map(|n| (1 << n)) {
            for constrain_num in 0..=2 {
                println!(
                    "compile unit_size: {}, constrain_num: {}",
                    unit_size, constrain_num
                );
                let key = SchemeKey {
                    unit_size,
                    constrain_num,
                    block_array_size: 5,
                    grid_array_size: 5,
                };
                let (ptx, log) = Ptx::compile(format_code(key), dev.compute_capability());
                let Ok(ptx) = ptx else { panic!("{log}") };
                ptxs.insert(key, ptx);
            }
        }

        // 打印所有编译好的kernel信息
        gpu.apply(|ctx| {
            println!("{}", ctx.dev().info());
            for (key, ptx) in ptxs.iter() {
                println!("{:?}", key);
                println!(
                    "unit_size: {}, constrain_num: {}\n{}",
                    key.unit_size,
                    key.constrain_num,
                    ctx.load(&ptx)
                        .get_kernel(CString::new(kernel_name(key)).unwrap())
                        .info()
                );
                println!("----------------------------------------");
            }
        });
    }

    fn copute_with_check<const N: usize, const TRANS_N: usize>(
        gpu: &Gpu,
        dev: &Device,
        shape: [usize; N],
    ) -> Duration {
        assert!(TRANS_N <= N, "TRANS_N must be less than or equal to N");

        use super::Rearrange;
        use cuda::memcpy_d2h;
        use mem_rearrange::Rearranging as CpuRearrange;
        use rand::Rng;
        use tensor::digit_layout::types::U64;
        use tensor::ndarray_layout::{ArrayLayout, Endian::BigEndian};

        let dt = U64;

        let mut r_shape = shape;
        r_shape[0..TRANS_N].reverse();

        let trans_param: [usize; TRANS_N] =
            (0..TRANS_N).rev().collect::<Vec<_>>().try_into().unwrap();

        let mut src = vec![0u64; shape.iter().product::<usize>()];
        rand::rng().fill(&mut src[..]);

        let ele = dt.nbytes();
        let s_src = ArrayLayout::<N>::new_contiguous(&shape, BigEndian, ele);
        let s_dst =
            ArrayLayout::<N>::new_contiguous(&r_shape, BigEndian, ele).transpose(&trans_param);

        let shemed_rearrange = Rearrange::new(dev)
            .scheme(dev, &s_dst, &s_src, ele)
            .unwrap();

        let (dst_ans, time) = gpu.apply(|ctx| {
            let stream = ctx.stream();

            let src = stream.from_host(&src);
            let mut dst = stream.malloc::<u8>(src.len());

            let start_event = stream.record();

            let mut modules = HashMap::new();

            stream.bench(
                |_, stream| {
                    shemed_rearrange.launch(
                        dst.as_mut_ptr().cast(),
                        src.as_ptr().cast(),
                        stream,
                        &mut modules,
                    );
                },
                5,
                1,
            );

            let end_event = stream.record();
            end_event.synchronize();
            let time = end_event.elapse_from(&start_event);

            let mut host = vec![0u64; shape.iter().product::<usize>()];
            memcpy_d2h(&mut host, &dst);
            (host, time)
        });

        let mut dst_ref = vec![0u64; shape.iter().product::<usize>()];

        unsafe {
            CpuRearrange::new(&s_dst, &s_src, ele)
                .unwrap()
                .launch(dst_ref.as_mut_ptr().cast(), src.as_ptr().cast());
        }

        assert_eq!(dst_ans, dst_ref);
        time
    }

    #[test]
    fn test_compute() {
        assert!(cuda::init().is_ok());
        let dev = Device::new(0);
        let gpu = Gpu::new(dev.context(), Default::default());

        let shape = [2];
        let time = copute_with_check::<1, 1>(&gpu, &dev, shape);
        println!("time: {time:?}");

        let shape = [13];
        let time = copute_with_check::<1, 1>(&gpu, &dev, shape);
        println!("time: {time:?}");

        let shape = [16, 2, 16];
        let time = copute_with_check::<3, 3>(&gpu, &dev, shape);
        println!("time: {time:?}");

        let shape = [32, 2, 17];
        let time = copute_with_check::<3, 3>(&gpu, &dev, shape);
        println!("time: {time:?}");

        let shape = [32, 2, 17, 2, 13];
        let time = copute_with_check::<5, 5>(&gpu, &dev, shape);
        println!("time: {time:?}");
    }
}
