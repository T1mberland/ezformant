use ndarray::prelude::*;
use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

pub fn pre_emphasis(signal: &mut [f64], alpha: f64) {
    for i in (1..signal.len()).rev() {
        signal[i] -= alpha * signal[i - 1];
    }
    signal[0] *= 1.0 - alpha;
}

/// Computes the autocorrelation of a signal up to a specified lag.
/// 
/// # Arguments
/// 
/// * `signal` - A slice of f64 representing the input signal.
/// * `lag` - The maximum lag for which to compute autocorrelation.
/// 
/// # Returns
/// 
/// A vector containing autocorrelation values from lag 0 to `lag`.
fn autocorrelate(signal: &[f64], lag: usize) -> Vec<f64> {

}

/// Computes the dot product of two slices.
/// 
/// # Arguments
/// 
/// * `a` - A slice of f64.
/// * `b` - A slice of f64.
/// 
/// # Returns
/// 
/// The dot product as an f64.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Reverses a slice and returns a new vector.
/// 
/// # Arguments
/// 
/// * `a` - A slice of f64.
/// 
/// # Returns
/// 
/// A new vector containing the elements of `a` in reverse order.
fn reverse_slice(a: &[f64]) -> Vec<f64> {
    let mut reversed = a.to_vec();
    reversed.reverse();
    reversed
}

/// Implements the Levinson-Durbin recursion algorithm.
/// 
/// # Arguments
/// 
/// * `signal` - A slice of f64 representing the input signal.
/// * `order` - The order of the recursion.
/// * `r` - An optional slice of f64 representing the autocorrelation coefficients.
/// 
/// # Returns
/// 
/// A tuple containing:
/// - A vector of filter coefficients (`a`).
/// - The final prediction error (`E`).
pub fn levinson(signal: &[f64], order: usize, r: Option<&[f64]>) -> (Vec<f64>, f64){

}

/// Compute the frequency response of the LPC filter
pub fn compute_frequency_response(
    lpc_coeffs: &[f64], 
    sample_rate: f64, 
    num_points: usize
) -> Vec<(f64, f64)> {
    let mut response = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let freq = i as f64 / num_points as f64 * sample_rate / 2.0; // Frequency in Hz
        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
        let z = Complex::new(omega.cos(), -omega.sin()); // Complex exponential
        let denominator: Complex<f64> = lpc_coeffs
            .iter()
            .enumerate()
            .map(|(k, &a_k)| z.powi(-(k as i32)) * a_k)
            .sum();
        let h = Complex::new(1.0, 0.0) / denominator; 
        response.push((freq, h.norm()));
    }
    response
}

