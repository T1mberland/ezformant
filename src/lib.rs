use rustfft::{FftPlanner, num_complex::Complex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn basic_fft() {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(2048);

    let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32}; 2048];
    for i in 0..2048 {
        buffer[i] = Complex { re: f32::cos(i as f32), im: 0.0f32 };
    }

    fft.process(&mut buffer);
    
    for i in 0..2048 {
        println!("{}", buffer[i].re);
    }
}

