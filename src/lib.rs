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
pub fn lpc_filter_freq_response(
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

    // In `lpc_filter_freq_responce` before autocorrelation
    lpc::pre_emphasis(&mut data, 0.97);

    let r = lpc::autocorrelate(&data, lpc_order);

    match lpc::levinson(lpc_order, &r) {
        (a,_e) => {
            lpc::compute_frequency_response(&a, sample_rate, num_points)
                .into_iter()
                .map(|(_, mag)| mag)
                .collect()
        }
    }
}


// Returns [F1, F2, F3, F4, LPC_frequency_response]
#[wasm_bindgen]
pub fn lpc_filter_freq_response_with_peaks(
    mut data: Vec<f64>, 
    lpc_order: usize, 
    sample_rate: f64,
    num_points: usize
) -> Vec<f64> {
    const FORMANT_NUM: usize = 4;

    // Subtract the mean to make the signal zero-mean
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    for sample in data.iter_mut() {
        *sample -= mean;
    }

    // Apply windowing (e.g., Hamming window)
    for i in 0..data.len() {
        data[i] *= 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (data.len() as f64 - 1.0)).cos();
    }

    lpc::pre_emphasis(&mut data, 0.97);

    let r = lpc::autocorrelate(&data, lpc_order);
    let (lpc_coeff, _) = lpc::levinson(lpc_order, &r);
    let formants = lpc::formant_detection(&lpc_coeff, sample_rate);
    //let formants = vec![0.0; FORMANT_NUM];
    let lpc_freq_response: Vec<f64> =
            lpc::compute_frequency_response(&lpc_coeff, sample_rate, num_points)
                .into_iter()
                .map(|(_, mag)| mag)
                .collect();

    let mut result = Vec::with_capacity(FORMANT_NUM + lpc_freq_response.len());
    for i in 0..FORMANT_NUM {
        if i < formants.len() {
            result.push(formants[i]);
        } else {
            result.push(0.0);
        }
    }
    result.extend(&lpc_freq_response);

    result
}

// returns [F1,f2,f3,f4]
#[wasm_bindgen]
pub fn formant_detection(
    mut data: Vec<f64>, 
    lpc_order: usize, 
    sample_rate: f64,
) -> Vec<f64> {
    const FORMANT_NUM: usize = 4;

    // Subtract the mean to make the signal zero-mean
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    for sample in data.iter_mut() {
        *sample -= mean;
    }

    // Apply windowing (e.g., Hamming window)
    for i in 0..data.len() {
        data[i] *= 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (data.len() as f64 - 1.0)).cos();
    }

    lpc::pre_emphasis(&mut data, 0.97);

    let r = lpc::autocorrelate(&data, lpc_order);
    let (lpc_coeff, _) = lpc::levinson(lpc_order, &r);
    let formants = lpc::formant_detection(&lpc_coeff, sample_rate);

    formants
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
        let (l, _) = lpc::levinson(3, &r);
        for i in 0..(l.len()) { 
            //println!("l[i]={}, when it should be {}", l[i], y4[i]);
            assert!((l[i] - y4[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn formant_detection_test() {
        let lpc = [  1.        , -1.75325333,  1.97953403, -1.80343314,  1.20047156,
                     0.00740131, -0.46918192,  0.74669944, -0.81144139,  0.5992474 ,
                    -0.22257812,  0.12155728,  0.04168977];
        let fs = 11025.0f64; // sampling rate
        let peaks = lpc::formant_detection(&lpc, fs);
        let answers = [654.0, 1131.0, 2382.0, 2826.0, 3539.0];
        let epsilon = 10.0;

        let mut check = [false; 5];
        for peak in peaks {
            let mut check2 = false;
            for i in 0..5 {
                check[i] = check[i] || ((peak - answers[i]).abs() < epsilon);
                check2 = check2 || (peak - answers[i]).abs() < epsilon;
            }
  
            assert!(check2);
        }
        
        for c in check { assert!(c); }
    }

    #[test]
    fn peak_detection_test() {
        let lpc = [  1.        , -1.75325333,  1.97953403, -1.80343314,  1.20047156,
                     0.00740131, -0.46918192,  0.74669944, -0.81144139,  0.5992474 ,
                    -0.22257812,  0.12155728,  0.04168977];
        let fs = 11025.0f64; // sampling rate
        let peaks = lpc::peak_detection(&lpc, fs);

        const PEAKS_NUM: usize = 11;
        let epsilon = 10.0;
        let answers: [f64; PEAKS_NUM] = [654.0, 1131.0, 2382.0, 2826.0, 3539.0, -5512.0, -2826.0, -3539.0, -2382.0, -1131.0, -654.0];

        let mut check: [bool; PEAKS_NUM] = [false; PEAKS_NUM];
        for peak in peaks {
            let mut check2 = false;
            for i in 0..PEAKS_NUM {
                check[i] = check[i] || ((peak - answers[i]).abs() < epsilon);
                check2 = check2 || (peak - answers[i]).abs() < epsilon;
            }
    
            assert!(check2);
        }
        
        for c in check { assert!(c); }
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


