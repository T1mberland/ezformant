use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

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

    let mut downsampled: [f64; FRAME_LENGTH] = [0.0; FRAME_LENGTH];
    downsampler(&data, &mut downsampled, 4);
    let r = autocorrelate(&downsampled, FIXED_LPC_ORDER);

    
    // Solver bench
    /*
    c.bench_function("bench test", |b| b.iter(|| {
        // Test 1
        //formant_detection(data.clone(), FIXED_LPC_ORDER, 441000f64);

        
        // Tets2: no change
        formant_detection_fixed(data.clone(), 441000.0f64);
    }));
    */

    c.bench_function("levinson, test", |b| b.iter(|| {

        //levinson(FIXED_LPC_ORDER, &r);
        levinson2(FIXED_LPC_ORDER, &r); // very fast

    }));
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
/// * `order` - The order of the recursion.
/// * `r` - An optional slice of f64 representing the autocorrelation coefficients.
/// 
/// # Returns
/// 
/// A tuple containing:
/// - A vector of filter coefficients (`a`).
/// - The final prediction error (`E`).
fn levinson(order: usize, r: &[f64]) -> (Vec<f64>, f64){
    let p = order;

    if p == 0 {
        return (vec![1.0], r[0]);
    } else if p == 1 {
        let a1 = -r[1]/r[0];
        return (vec![1.0, a1], r[0] + r[1]*a1);
    }


    let (aa, ee) = levinson(p-1, r);

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

criterion_group!(benches, criterion_bench);
criterion_main!(benches);


