use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root, include_cuda};

    assert!(
        find_cuda_root().is_some(),
        "cuda not found, check $CUDA_ROOT env var"
    );
    let nccl = Cfg::new("nccl");
    if find_nccl_root().is_some() {
        nccl.define()
    }

    // 为 random_sample 生成 bindings

    // 获取cuda根目录
    let cuda_root = find_cuda_root().unwrap();

    include_cuda();

    // 获取输出目录
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // 获取当前目录
    let current_dir = env::current_dir().unwrap();

    let cuda_src_dir = "src/op/random_sample/";
    // 定义CUDA源文件路径
    let cuda_src_path = current_dir.join(cuda_src_dir).join("sample.cu");
    let header_path = current_dir.join(cuda_src_dir).join("sample.h");

    // 输出库文件路径 - 更改为明确的库名称
    let lib_name = "random_sample";
    let lib_output_path = out_path.join(format!("lib{}.so", lib_name));

    // 运行nvcc编译命令
    let status = Command::new(cuda_root.join("bin/nvcc"))
        .arg(cuda_src_path)
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-shared")
        .arg("-o")
        .arg(&lib_output_path)
        .arg("-Wno-deprecated-gpu-targets")
        .status()
        .expect("无法执行nvcc命令");

    if !status.success() {
        panic!("nvcc编译失败");
    }

    // 重新运行构建脚本的条件
    println!("cargo:rerun-if-changed={}/sample.cu", cuda_src_dir);
    println!("cargo:rerun-if-changed={}/sample.cuh", cuda_src_dir);
    // 打印cargo相关信息，告诉Rust编译器动态库的位置
    println!("cargo:rustc-link-search=native={}", out_dir);
    // 使用正确的库名称
    println!("cargo:rustc-link-lib=dylib={}", lib_name);

    // 使用bindgen生成Rust绑定
    let bindings = bindgen::Builder::default()
        // 指定头文件
        .header(header_path.to_str().unwrap())
        // 告诉bindgen在哪里可以找到CUDA头文件
        .clang_arg(format!("-I{}", cuda_root.join("include").to_str().unwrap()))
        .clang_arg(format!(
            "-I{}",
            current_dir.join(cuda_src_dir).to_str().unwrap()
        ))
        .clang_arg("-x")
        .clang_arg("c++")
        // 只生成特定函数
        .allowlist_function("calculate_workspace_size_half")
        .allowlist_function("argmax_half")
        .allowlist_function("sample_half")
        // 使用Rust风格的枚举
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // 使用derive特性
        .derive_default(true)
        .derive_debug(true)
        // 处理cargo回调
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // 生成绑定
        .generate()
        // 处理错误
        .expect("无法生成绑定");

    // 将绑定写入指定的文件
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("无法写入绑定");
}
