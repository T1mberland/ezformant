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

    /*
    let autocorr = lpc::autocorrelation_time_domain(&data, lpc_order);
    // Log the entire autocorrelation sequence for debugging
    log("Autocorrelation Sequence:");
    for (i, val) in autocorr.iter().enumerate() {
        log(&format!("r[{}] = {}", i, val));
    }
    */

    match lpc::levinson(&data, lpc_order, None) {
        (a,_e) =>
            lpc::compute_frequency_response(&a, sample_rate, num_points)
                .into_iter()
                .map(|(_, mag)| mag)
                .collect()
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    //#[test]
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

