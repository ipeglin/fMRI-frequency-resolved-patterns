// git2 fix for Windows
fn main() {
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=advapi32");
}
