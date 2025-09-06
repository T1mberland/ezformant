use ezformant::*;
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use wasm_bindgen::prelude::*;

// mod lpc;

#[wasm_bindgen]
pub fn process_audio(data: Vec<f32>) -> Vec<f32> {
    let len = data.len();
    let mut fft_input: Vec<Complex<f32>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(len);

    fft.process(&mut fft_input);

    let half_len = len / 2;
    fft_input
        .iter()
        .take(half_len)
        .map(|x| x.abs() + 1e-10)
        .collect()
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_many(a: &str, b: &str);
}

// ------------------
// Helper Functions
// ------------------

/// Downsample the input signal by the given factor.
fn downsample(input: &[f64], factor: usize) -> Vec<f64> {
    input.iter().step_by(factor).copied().collect()
}

/// Subtract the mean from the input data (in-place).
fn subtract_mean_in_place(data: &mut [f64]) {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    for sample in data.iter_mut() {
        *sample -= mean;
    }
}

/// Apply a Hamming window to the input data (in-place).
fn apply_hamming_window_in_place(data: &mut [f64]) {
    let n = data.len() as f64;
    // Avoid degeneracy (e.g., 0-length array).
    if data.is_empty() {
        return;
    }
    for (i, sample) in data.iter_mut().enumerate() {
        let ratio = i as f64 / (n - 1.0);
        *sample *= 0.54 - 0.46 * (2.0 * std::f64::consts::PI * ratio).cos();
    }
}

/// Apply pre-emphasis filter to the input data (in-place).
/// `alpha` is the pre-emphasis coefficient (commonly around 0.95–0.97).
fn pre_emphasize_in_place(data: &mut [f64], alpha: f64) {
    lpc::pre_emphasis(data, alpha);
}

/// Preprocess signal by:
/// 1) subtracting the mean,
/// 2) applying a Hamming window,
/// 3) applying pre-emphasis.
fn preprocess_signal(data: &mut [f64], alpha: f64) {
    subtract_mean_in_place(data);
    apply_hamming_window_in_place(data);
    pre_emphasize_in_place(data, alpha);
}

// ------------------
// Public API
// ------------------

#[wasm_bindgen]
pub fn lpc_filter_freq_response_with_downsampling(
    original_data: Vec<f64>,
    lpc_order: usize,
    original_sample_rate: f64,
    downsample_factor: usize,
    num_points: usize,
) -> Vec<f64> {
    // Downsample
    let mut data = downsample(&original_data, downsample_factor);
    let sample_rate = original_sample_rate / downsample_factor as f64;

    // Preprocess signal
    preprocess_signal(&mut data, 0.97);

    // Compute autocorrelation
    let r = lpc::autocorrelate(&data, lpc_order);

    // Solve for LPC coefficients
    let (a, _e) = lpc::levinson(lpc_order, &r);

    // Compute frequency response
    lpc::compute_frequency_response(&a, sample_rate, num_points)
        .into_iter()
        .map(|(_, mag)| mag)
        .collect()
}

#[wasm_bindgen]
pub fn lpc_filter_freq_response(
    mut data: Vec<f64>,
    lpc_order: usize,
    sample_rate: f64,
    num_points: usize,
) -> Vec<f64> {
    // Preprocess signal
    preprocess_signal(&mut data, 0.97);

    // Compute autocorrelation
    let r = lpc::autocorrelate(&data, lpc_order);

    // Solve for LPC coefficients
    let (a, _e) = lpc::levinson(lpc_order, &r);

    // Compute frequency response
    lpc::compute_frequency_response(&a, sample_rate, num_points)
        .into_iter()
        .map(|(_, mag)| mag)
        .collect()
}

// Returns [F1, F2, F3, F4, LPC_frequency_response]
#[wasm_bindgen]
pub fn lpc_filter_freq_response_with_peaks(
    mut data: Vec<f64>,
    lpc_order: usize,
    sample_rate: f64,
    num_points: usize,
) -> Vec<f64> {
    const FORMANT_NUM: usize = 4;

    // Preprocess signal
    preprocess_signal(&mut data, 0.97);

    // Compute autocorrelation
    let r = lpc::autocorrelate(&data, lpc_order);
    let (lpc_coeff, _) = lpc::levinson(lpc_order, &r);

    // Detect formants
    let formants = lpc::formant_detection(&lpc_coeff, sample_rate);

    // Compute LPC frequency response
    let lpc_freq_response: Vec<f64> =
        lpc::compute_frequency_response(&lpc_coeff, sample_rate, num_points)
            .into_iter()
            .map(|(_, mag)| mag)
            .collect();

    // Prepare result
    let mut result = Vec::with_capacity(FORMANT_NUM + lpc_freq_response.len());

    // Fill in up to FORMANT_NUM formants
    for i in 0..FORMANT_NUM {
        result.push(*formants.get(i).unwrap_or(&0.0));
    }
    // Then append the frequency response
    result.extend(&lpc_freq_response);

    result
}

// returns [F1,f2,f3,f4]
#[wasm_bindgen]
pub fn formant_detection(mut data: Vec<f64>, lpc_order: usize, sample_rate: f64) -> Vec<f64> {
    // Preprocess signal
    preprocess_signal(&mut data, 0.97);

    // Compute autocorrelation
    let r = lpc::autocorrelate(&data, lpc_order);

    // Solve for LPC coefficients
    let (lpc_coeff, _) = lpc::levinson(lpc_order, &r);

    // Detect formants
    lpc::formant_detection(&lpc_coeff, sample_rate)
}

// returns [F1,f2,f3,f4]
#[wasm_bindgen]
pub fn formant_detection_with_downsampling(
    original_data: Vec<f64>,
    lpc_order: usize,
    original_sample_rate: f64,
    downsample_factor: usize,
) -> Vec<f64> {
    // Downsample
    let mut data = downsample(&original_data, downsample_factor);
    let sample_rate = original_sample_rate / downsample_factor as f64;

    // Preprocess signal
    preprocess_signal(&mut data, 0.97);

    // Compute autocorrelation
    let r = lpc::autocorrelate(&data, lpc_order);

    // Solve for LPC coefficients
    let (lpc_coeff, _) = lpc::levinson(lpc_order, &r);

    // Detect formants
    lpc::formant_detection(&lpc_coeff, sample_rate)
}

// ------------------
// Tests
// ------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper function to manually downsample the data.
    fn manual_downsample(data: &[f64], factor: usize) -> Vec<f64> {
        data.iter().step_by(factor).cloned().collect()
    }

    #[test]
    fn test_lpc_filter_freq_response_with_downsampling() {
        // Parameters for the synthetic test signal
        let original_sample_rate = 16000.0; // in Hz
        let frequency = 440.0; // A4 note in Hz
        let duration = 1.0; // in seconds
        let num_samples = (original_sample_rate * duration) as usize;

        // Generate a sine wave as the original data
        let original_data: Vec<f64> = (0..num_samples)
            .map(|n| (2.0 * PI * frequency * (n as f64) / original_sample_rate).sin())
            .collect();

        // LPC parameters
        let lpc_order = 10;
        let downsample_factor = 2;
        let num_points = 512;

        // Call the function that includes downsampling
        let response_with_downsampling = lpc_filter_freq_response_with_downsampling(
            original_data.clone(),
            lpc_order,
            original_sample_rate,
            downsample_factor,
            num_points,
        );

        // Manually downsample the original data
        let downsampled_data = manual_downsample(&original_data, downsample_factor);
        let downsampled_sample_rate = original_sample_rate / (downsample_factor as f64);

        // Call the function without downsampling
        let response_without_downsampling = lpc_filter_freq_response(
            downsampled_data.clone(),
            lpc_order,
            downsampled_sample_rate,
            num_points,
        );

        // Define an acceptable error tolerance
        let epsilon = 1e-3;

        // Ensure both responses have the same number of points
        assert_eq!(
            response_with_downsampling.len(),
            response_without_downsampling.len(),
            "Frequency responses have different lengths"
        );

        // Compare each point in the frequency responses
        for (i, (resp_ds, resp)) in response_with_downsampling
            .iter()
            .zip(response_without_downsampling.iter())
            .enumerate()
        {
            let diff = (resp_ds - resp).abs();
            if diff > epsilon {
                panic!(
                    "Frequency response differs at index {}: with_downsampling = {}, without_downsampling = {}, difference = {} exceeds epsilon = {}",
                    i, resp_ds, resp, diff, epsilon
                );
            }
        }
    }
}
