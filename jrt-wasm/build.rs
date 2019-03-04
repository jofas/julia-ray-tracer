extern crate webgl_generator;

use std::fs::File;
use webgl_generator::*;

fn main() {
    let mut file = File::create("src/webgl2_context.rs")
      .unwrap();

    Registry::new(Api::WebGl2, Exts::ALL)
        .write_bindings(StdwebGenerator, &mut file)
        .unwrap();
}
