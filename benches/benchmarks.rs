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

fn criterion_bench(c: &mut Criterion) {
    let data = read_audio_data();

    c.bench_function("bench test", |b| b.iter(|| {
        //formant_detection(data.clone(), 16, 441000f64);
        formant_detection_fixed(&data[..2048], 16, 441000f64);
    }));
}

fn peak_detection_fixed(lpc_coeffs: &[f64; 2048], sample_rate: f64) -> Vec<f64> {
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

pub fn formant_detection_fixed(lpc_coeffs: &[f64; 2048], sample_rate: f64) -> Vec<f64> {
    let peaks = peak_detection_fixed(lpc_coeffs, sample_rate);
    let mut formants = Vec::with_capacity(peaks.len());

    for peak in peaks {
        if peak < 10.0 || peak > (sample_rate/2.0 - 10.0) { continue; }

        formants.push(peak);
    }

    formants.sort_by(|a, b| a.partial_cmp(b).unwrap());

    formants
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);


