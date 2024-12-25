use std::f32::EPSILON;

use ndarray::prelude::*;
use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;
use aberth::aberth;
use aberth::AberthSolver;

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
/// * `maxlag` - The maximum lag for which to compute autocorrelation.
/// 
/// # Returns
/// 
/// A vector containing autocorrelation values from lag 0 to `lag`.
pub fn autocorrelate(signal: &[f64], maxlag: usize) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    let n = signal.len();

    for lag in 0..=maxlag {
        let mut r = 0.0;
        for i in 0..n {
            if i + lag < n {
                r += signal[i] * signal[i+lag];
            } else {
                break;
            }
        }

        result.push(r);
    }

    result
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
pub fn levinson(signal: &[f64], order: usize, r: &[f64]) -> (Vec<f64>, f64){
    let p = order;

    if p == 0 {
        return (vec![1.0], r[0]);
    } else if p == 1 {
        let a1 = -r[1]/r[0];
        return (vec![1.0, a1], r[0] + r[1]*a1);
    }


    let (aa, ee) = levinson(signal, p-1, r);

    let mut k = 0.0;
    for j in 0..p {
        //println!("levinson: j={}", j);
        k += aa[j]*r[p-j];
    }
    k = - k / ee;

    let e = ee*(1.0 - k*k);

    let mut u = aa.clone();
    u.push(0.0);

    let mut v = u.clone();
    v.reverse();

    let result = u.iter().enumerate().map(|(ix, &uu)| uu + k*v[ix]).collect();

    (result, e)
}

/// Implements the Levinson-Durbin recursion algorithm iteratively.
/// 
/// # Arguments
/// 
/// * `signal` - (Optional) A slice of f64 representing the input signal. 
///    You can ignore it here if autocorrelation `r` is precomputed externally.
/// * `order`  - The order of the recursion (filter).
/// * `r`      - A slice of f64 representing the autocorrelation coefficients.
///              Must have length >= `order + 1`.
/// 
/// # Returns
/// 
/// A tuple containing:
/// - A vector of filter coefficients `[a0, a1, ..., a_order]` (with `a0 = 1.0`).
/// - The final prediction error (`E`).
pub fn levinson2(signal: &[f64], order: usize, r: &[f64]) -> (Vec<f64>, f64) {
    // We'll store the filter coefficients in `a`.
    // a[0] is always 1.0 by definition.
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;

    // The initial prediction error is the zero-lag autocorrelation.
    let mut e = r[0];

    // Iteratively compute the filter coefficients for each order i = 1..=order
    for i in 1..=order {
        // Calculate the reflection (Parcor) coefficient, often called `k` or `lambda`.
        let mut lambda = 0.0;
        for j in 0..i {
            lambda += a[j] * r[i - j];
        }
        lambda = -lambda / e;

        // Update the coefficients a[0..=i]. 
        // We only need to update up to i//2 indices in-place, mirroring around the midpoint.
        for j in 0..=((i) / 2) {
            let temp = a[j] + lambda * a[i - j];
            a[i - j] += lambda * a[j];
            a[j] = temp;
        }

        // Update the prediction error.
        e *= 1.0 - lambda * lambda;
    }

    (a, e)
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
            .map(|(k, &a_k)| z.powi(k as i32) * a_k)
            .sum();
        let h = Complex::new(1.0, 0.0) / denominator; 
        response.push((freq, h.norm()));
    }
    response
}

pub fn peak_detection(lpc_coeffs: &[f64], sample_rate: f64) -> Vec<f64> {
    const EPSILON: f64 = 0.001;
    const MAX_ITERATIONS: u32 = 15;
    let mut solver = AberthSolver::new();
    solver.epsilon = EPSILON;
    solver.max_iterations = MAX_ITERATIONS;

    let roots = solver.find_roots(lpc_coeffs).to_vec();
    let mut peaks = Vec::with_capacity(roots.len());

    for root in roots {
        let theta = root.arg();
        if 0.0 <= theta {
            peaks.push(theta * sample_rate / 2.0 / std::f64::consts::PI);
        } else if - std::f64::consts::PI <= theta && theta < 0.0 {
            peaks.push((theta + 2.0 * std::f64::consts::PI) * sample_rate / 2.0 / std::f64::consts::PI);
        } else {
            // Won't happen
        }
    }

    peaks
}

pub fn formant_detection(lpc_coeffs: &[f64], sample_rate: f64) -> Vec<f64> {
    let peaks = peak_detection(lpc_coeffs, sample_rate);
    let mut formants = Vec::with_capacity(peaks.len());

    for peak in peaks {
        if peak < 10.0 || peak > (sample_rate/2.0 - 10.0) { continue; }

        formants.push(peak);
    }

    formants.sort_by(|a, b| a.partial_cmp(b).unwrap());

    formants
}


/* ------------------------------------------ */
/* ------------------------------------------ */
/* ------------------------------------------ */


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levinson_durbin() {
        let r = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        let order = 5;

        let (a, e) = levinson2(&[], order, &r);
        let (expected_a, expected_e) = levinson(&[], order, &r);

        let epsilon = 1e-4;

        for (computed, expected) in a.iter().zip(expected_a.iter()) {
            assert!((computed - expected).abs() < epsilon, "Coefficient mismatch: got {}, expected {}", computed, expected);
        }

        assert!((e - expected_e).abs() < epsilon, "Error mismatch: got {}, expected {}", e, expected_e);
    }
}
