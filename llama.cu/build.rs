use build_script_cfg::Cfg;
use search_cuda_tools::{find_cuda_root, find_nccl_root};
use std::{env, path::PathBuf, process::Command};

fn main() {
    let Some(cuda_root) = find_cuda_root() else {
        panic!("cuda not found, check $CUDA_ROOT env var")
    };
    let nccl = Cfg::new("nccl");
    if find_nccl_root().is_some() {
        nccl.define()
    }

    let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());
    let proj_dir = PathBuf::from(&env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let src_dir = proj_dir.join("src/op/random_sample");
    let lib_name = "random_sample";
    let header = src_dir.join("sample.h");
    let source = src_dir.join("sample.cu");

    let status = Command::new(cuda_root.join("bin/nvcc"))
        .arg(source)
        .args(["-Xcompiler", "-fPIC"])
        .arg("-shared")
        .arg("-o")
        .arg(out_dir.join(format!("lib{lib_name}.so")))
        .status()
        .expect("nvcc command failed");

    if !status.success() {
        panic!("nvcc compile failed")
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib={lib_name}");

    bindgen::Builder::default()
        .header(header.display().to_string())
        .clang_arg(format!("-I{}", src_dir.display()))
        .clang_arg(format!("-I{}", cuda_root.join("include").display()))
        .allowlist_function("calculate_workspace_size_half")
        .allowlist_function("argmax_half")
        .allowlist_function("sample_half")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .use_core()
        .derive_default(true)
        .derive_debug(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
