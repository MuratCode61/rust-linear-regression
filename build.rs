fn main(){
    println!(r"cargo:rustc-link-search=/usr/local/Cellar/gcc/10.2.0/lib/gcc/10");
    println!(r"cargo:rustc-link-search=/usr/local/opt/openblas/lib");

    // /usr/local/opt/lapack/lib
}