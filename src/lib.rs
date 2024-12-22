use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;

mod lpc;

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
pub fn lpc_filter_freq_responce(data: Vec<f64>, lpc_order: usize, sample_rate: f64, num_points: usize) -> Vec<f64> {
    //let autocorr = lpc::autocorrelation_frequency_domain(&data, lpc_order);
    let autocorr = lpc::autocorrelation_time_domain(&data, lpc_order);
    let mut lpc_coeffs = vec![0.0; lpc_order + 1];

    log("aut corr");
    log(&autocorr[0].to_string());
    log(&autocorr[1].to_string());
    log(&autocorr[2].to_string());
    log(&autocorr[3].to_string());
    log(&autocorr[4].to_string());
    log(&autocorr.len().to_string());

    match lpc::levinson(&autocorr, &mut lpc_coeffs) {
        Ok(()) => {
            lpc::compute_frequency_response(&lpc_coeffs, sample_rate, num_points)
                .into_iter()
                .map(|(x,_)| x)
                .collect()
        }
        Err(e) => {
            log(&e);
            vec![0.0; lpc_order + 1]
        }
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

    #[test]
    fn test2() {
        lpc::lpctest();
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

