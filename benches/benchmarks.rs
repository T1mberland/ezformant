use serde_json::{Result, Value, Number};
use serde::Deserialize;
use std::{f32::EPSILON, fs};
use std::path::PathBuf;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use ezformant::*;

use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};
use wasm_bindgen::prelude::*;
use aberth::aberth;
use aberth::AberthSolver;

const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

#[derive(Debug, Deserialize)]
struct JsonData(HashMap<String, f64>);

fn read_audio_data() -> Vec<f64> {
    let mut path = PathBuf::from(CARGO_MANIFEST_DIR);
    path.push("benches/audio_frame.json");
    
    let json_string: String = fs::read_to_string(path).expect("Failed to read data file");
    let json_data: JsonData = serde_json::from_str(&json_string).unwrap();

    let mut data: Vec<f64> = vec![0.0; 2048];

    for (key, val) in &json_data.0 {
        let ix = key.parse::<usize>().unwrap();
        data[ix] = *val;
    }

    data
}

fn downsampler(input: &[f64], output: &mut [f64], factor: usize) {
    for (i, value) in input.iter().step_by(factor).enumerate() {
        output[i] = *value;
    }
}

const FIXED_LPC_ORDER: usize = 14;

fn criterion_bench(c: &mut Criterion) {
    let data = read_audio_data();
    const FRAME_LENGTH: usize = 2048;
    
    // Solver bench
    c.bench_function("bench test", |b| b.iter(|| {
        // Test 1
        //formant_detection(data.clone(), FIXED_LPC_ORDER, 441000f64);

        
        // Tets2: no change
        formant_detection_fixed(data.clone(), 441000.0f64);
    }));

    /*
    c.bench_function("bench test", |b| b.iter(|| {
        let mut downsampled: [f64; FRAME_LENGTH];
        let mut tmp: [f64; FIXED_LPC_ORDER+1] = [0.0;FIXED_LPC_ORDER+1];

        downsampler(&data, &mut downsampled, 4);

        let coeffs = lpc::levinson();

        for i in 0..=FIXED_LPC_ORDER {
            tmp[i] = data[i];
        }

        formant_detection_fixed(&tmp, 441000.0f64);
    }));
    */
}

fn pre_emphasis(signal: &mut [f64], alpha: f64) {
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
fn autocorrelate(signal: &[f64], maxlag: usize) -> Vec<f64> {
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
fn levinson(signal: &[f64], order: usize, r: &[f64]) -> (Vec<f64>, f64){
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


fn levinson2(order: usize, r: &[f64]) -> (Vec<f64>, f64){
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


fn peak_detection_fixed<const N:usize>(lpc_coeffs: &[f64; N], sample_rate: f64) -> Vec<f64> {
    const EPSILON: f64 = 0.001;
    const MAX_ITERATIONS: u32 = 15;
    let mut solver = AberthSolver::new();
    solver.epsilon = EPSILON;
    solver.max_iterations = MAX_ITERATIONS;

    let roots = aberth(&lpc_coeffs, MAX_ITERATIONS, EPSILON).to_vec();
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

fn _formant_detection_fixed<const N:usize>(lpc_coeffs: &[f64; N], sample_rate: f64) -> Vec<f64> {
    let peaks = peak_detection_fixed(lpc_coeffs, sample_rate);
    let mut formants = Vec::with_capacity(peaks.len());

    for peak in peaks {
        if peak < 10.0 || peak > (sample_rate/2.0 - 10.0) { continue; }

        formants.push(peak);
    }

    formants.sort_by(|a, b| a.partial_cmp(b).unwrap());

    formants
}


fn formant_detection_fixed(
    mut data: Vec<f64>, 
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

    pre_emphasis(&mut data, 0.97);

    let r = autocorrelate(&data, FIXED_LPC_ORDER);
    let (lpc_coeff, _) = levinson(&data, FIXED_LPC_ORDER, &r);
    
    let mut lpc_coeff_slice: [f64; FIXED_LPC_ORDER+1] = [0.0; FIXED_LPC_ORDER+1];

    for i in 0..=FIXED_LPC_ORDER {
        lpc_coeff_slice[i] = lpc_coeff[i];
    }

    let formants = _formant_detection_fixed(&lpc_coeff_slice, sample_rate);

    formants

}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);


