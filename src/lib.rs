use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;

mod lpc;

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
            x.abs() + 1e-10
        })
        .collect()

}

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // The `console.log` is quite polymorphic, so we can bind it with multiple
    // signatures. Note that we need to use `js_name` to ensure we always call
    // `log` in JS.
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);

    // Multiple arguments too!
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_many(a: &str, b: &str);
}

#[wasm_bindgen]
pub fn lpc_filter_freq_responce(
    mut data: Vec<f64>, 
    lpc_order: usize, 
    sample_rate: f64, 
    num_points: usize
) -> Vec<f64> {
    // Subtract the mean to make the signal zero-mean
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    for sample in data.iter_mut() {
        *sample -= mean;
    }

    // Optionally, apply windowing (e.g., Hamming window)
    for i in 0..data.len() {
        data[i] *= 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (data.len() as f64 - 1.0)).cos();
    }

    let r = lpc::autocorrelate(&data, lpc_order);

    // In `lpc_filter_freq_responce` before autocorrelation
    lpc::pre_emphasis(&mut data, 0.97);

    match lpc::levinson(&data, lpc_order, &r) {
        (a,_e) => {
            lpc::compute_frequency_response(&a, sample_rate, num_points)
                .into_iter()
                .map(|(_, mag)| mag)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn autocorrelate_test() {
        let x7 = vec![2.0,3.0,-1.0,-2.0,1.0,4.0,1.0];
        let y7 = vec![36.0, 11.0, -16.0, -7.0, 13.0, 11.0, 2.0];
        let r = lpc::autocorrelate(&x7, 6);
        for i in 0..(r.len()) {
            //println!("{}", r[i]);
            assert_eq!(r[i], y7[i]);
        }
    }

    #[test]
    fn levinson_test() {
        let x7 = vec![2.0,3.0,-1.0,-2.0,1.0,4.0,1.0];
        let y4 = vec![1.0, -0.69190537, 0.76150628, -0.34575153];
        let r = lpc::autocorrelate(&x7, 3);
        let (l, _) = lpc::levinson(&x7, 3, &r);
        for i in 0..(l.len()) { 
            //println!("l[i]={}, when it should be {}", l[i], y4[i]);
            assert!((l[i] - y4[i]).abs() < 1e-6);
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

