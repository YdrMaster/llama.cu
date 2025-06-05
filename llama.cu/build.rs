use build_script_cfg::Cfg;
use search_cuda_tools::{find_cuda_root, find_nccl_root};
use search_maca_tools::find_maca_root;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let nccl = Cfg::new("nccl");
    if let Some(maca_root) = find_maca_root() {
        nccl.define();
        compile_bind(
            maca_root,
            "sample.maca",
            "htgpu_llvm/bin/htcc",
            ["-x", "hpcc", "-fPIC"],
            Some("__MACA_ARCH__"),
        )
    } else if let Some(cuda_root) = find_cuda_root() {
        if find_nccl_root().is_some() {
            nccl.define()
        }
        compile_bind(
            cuda_root,
            "sample.cu",
            "bin/nvcc",
            ["-Xcompiler", "-fPIC"],
            None,
        )
    } else {
        panic!("cuda not found, check $CUDA_ROOT env var")
    }
}

fn compile_bind(
    toolkit: impl AsRef<Path>,
    src: &str,
    compiler: &str,
    compiler_args: impl IntoIterator<Item = &'static str>,
    define: Option<&str>,
) {
    let toolkit = toolkit.as_ref();

    let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());
    let proj_dir = PathBuf::from(&env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let src_dir = proj_dir.join("src/op/random_sample");
    let lib_name = "random_sample";
    let header = src_dir.join("sample.h");
    let source = src_dir.join(src);

    let status = Command::new(toolkit.join(compiler))
        .args(compiler_args)
        .arg("-shared")
        .arg(source)
        .arg("-o")
        .arg(out_dir.join(format!("lib{lib_name}.so")))
        .status()
        .expect("compiler executable error");

    if !status.success() {
        panic!("compile error")
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib={lib_name}");

    let mut builder = bindgen::Builder::default();
    if let Some(define) = define {
        builder = builder.clang_arg(format!("-D{define}"))
    }
    builder
        .header(header.display().to_string())
        .clang_arg(format!("-I{}", src_dir.display()))
        .clang_arg(format!("-I{}", toolkit.join("include").display()))
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
