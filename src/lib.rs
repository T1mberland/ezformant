use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[wasm_bindgen]
pub fn process_audio(data: Vec<f32>, lpc_order: usize) -> Vec<f32> {
    let len = data.len();
    let mut fft_input: Vec<Complex<f32>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(len);

    fft.process(&mut fft_input);

    let half_len = len / 2;
    fft_input.iter()
        .take(half_len)
        .map(|x| {
            (x.abs() + 1e-10)
        })
        .collect()

}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test1() {
        let mut planner = FftPlanner::<f32>::new();
        let len = 256;
        let fft = planner.plan_fft_forward(256);

        let mut buffer = vec![Complex{ re: 0.0, im: 0.0 }; 256];       

        fft.process(&mut buffer);

        for x in buffer {
            println!("{}", x);
        }
    }
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

