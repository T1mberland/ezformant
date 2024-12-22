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
    let n = signal.len();
    let mut autocorr = Vec::with_capacity(lag + 1);
    
    for l in 0..=lag {
        let mut sum = 0.0;
        for i in 0..(n - l) {
            sum += signal[i] * signal[i + l];
        }
        autocorr.push(sum);
    }
    
    autocorr
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
pub fn levinson(signal: &[f64], order: usize, r: Option<&[f64]>) -> (Vec<f64>, f64) {
    let p = order;
    
    if r.is_none() {
        // Compute autocorrelation up to lag p
        let autocorr = autocorrelate(signal, p);
        // Pass autocorrelation as r
        return levinson(signal, order, Some(&autocorr));
    }
    
    let r = r.unwrap();
    
    if p == 1 {
        // Base case
        let a = vec![1.0, -r[1] / r[0]];
        let e = a.iter().zip(&r[..2]).map(|(x, y)| x * y).sum();
        return (a, e);
    } else {
        // Recursive case
        let (aa, ee) = levinson(signal, p - 1, Some(r));
        // Compute the reflection coefficient kk
        let aa_reversed = reverse_slice(&aa);
        let r_segment = &r[1..=p];
        let numerator = dot_product(&aa, &r_segment);
        let kk = -numerator / ee;
        
        // Construct U by appending 0 to aa
        let mut u = aa.clone();
        u.push(0.0);
        
        // Construct V by reversing U
        let mut v = reverse_slice(&u);
        
        // Update filter coefficients
        let a: Vec<f64> = u.iter().zip(v.iter()).map(|(u_i, v_i)| u_i + kk * v_i).collect();
        
        // Update prediction error
        let e = ee * (1.0 - kk * kk);

        log(&e.to_string());
        
        return (a, e);
    }
}

/// Computes autocorrelation using the frequency-domain method.
pub fn autocorrelation_frequency_domain(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let n = signal.len();
    if max_lag >= n {
        panic!("max_lag must be less than the length of the signal");
    }

    // Set up FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert real input to complex
    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Forward FFT
    fft.process(&mut buffer);

    // Convert to power spectrum (|X[k]|^2)
    //  i.e., buffer[k] = X[k] * conj(X[k]) = |X[k]|^2 (real)
    for val in buffer.iter_mut() {
        let mag_sq = val.norm_sqr();
        *val = Complex::new(mag_sq, 0.0);
    }

    // Inverse FFT (of the power spectrum) gives us the autocorrelation
    ifft.process(&mut buffer);

    // Normalize by N
    let scale = 1.0 / n as f64;
    let autocorr_time_domain: Vec<f64> = buffer
        .iter()
        .map(|c| c.re * scale)
        .collect();

    // Return up to max_lag
    autocorr_time_domain[..=max_lag].to_vec()
}

pub fn autocorrelation_time_domain(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let n = signal.len();
    let mut autocorr = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        for i in lag..n {
            autocorr[lag] += signal[i] * signal[i - lag];
        }
    }
    autocorr
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
        let _denominator: Complex<f64> = lpc_coeffs
            .iter()
            .enumerate()
            .map(|(k, &a_k)| z.powi(-(k as i32)-1) * a_k)
            .sum();
        let denominator = _denominator + Complex::new(1.0,0.0);
        // Correct computation: H(z) = 1 / A(z)
        let h = Complex::new(1.0, 0.0) / denominator; 
        response.push((freq, h.norm()));
    }
    response
}

