fn main() {
    use find_cuda_helper::find_cuda_root;
    use std::{env, path::PathBuf, process::Command};

    let out_path = PathBuf::from(&env::var_os("OUT_DIR").unwrap());
    let cuda_root = find_cuda_root().unwrap();
    let proj_path = PathBuf::from(&env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let attention_dir = proj_path.join("src/op/nt-attention");
    let lib_name = "attention_kv";
    let attention_h = attention_dir.join("attention_kv_nh_64.h");
    let attention_c = attention_dir.join("attention_kv_nh_64.c");
    println!("cargo:rerun-if-changed={}", attention_c.display());

    let status = Command::new(cuda_root.join("bin/nvcc"))
        .arg(attention_c)
        .args(["-Xcompiler", "-fPIC"])
        .arg("-shared")
        .arg("-o")
        .arg(out_path.join(format!("lib{lib_name}.so")))
        .status()
        .expect("nvcc command failed");

    if !status.success() {
        panic!("nvcc compile failed")
    }

    println!("cargo:rustc-link-search={}", out_path.display());
    println!("cargo:rustc-link-lib={lib_name}");

    bindgen::Builder::default()
        .header(attention_h.display().to_string())
        .clang_arg(format!("-I{}", attention_dir.display()))
        .clang_arg(format!("-I{}", cuda_root.join("include").display()))
        .allowlist_function("launch_attention_kv.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .use_core()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!")
}
