use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};

pub mod lpc;

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

// ------------------
// Helper Functions
// ------------------

/// Downsample the input signal by the given factor.
pub fn downsample(input: &[f64], factor: usize) -> Vec<f64> {
    input.iter().step_by(factor).copied().collect()
}

/// Subtract the mean from the input data (in-place).
pub fn subtract_mean_in_place(data: &mut [f64]) {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    for sample in data.iter_mut() {
        *sample -= mean;
    }
}

/// Apply a Hamming window to the input data (in-place).
pub fn apply_hamming_window_in_place(data: &mut [f64]) {
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
pub fn pre_emphasize_in_place(data: &mut [f64], alpha: f64) {
    lpc::pre_emphasis(data, alpha);
}

/// Preprocess signal by:
/// 1) subtracting the mean,
/// 2) applying a Hamming window,
/// 3) applying pre-emphasis.
pub fn preprocess_signal(data: &mut [f64], alpha: f64) {
    subtract_mean_in_place(data);
    apply_hamming_window_in_place(data);
    pre_emphasize_in_place(data, alpha);
}

// ------------------
// Tests
// ------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn autocorrelate_test() {
        let x7 = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let y7 = vec![36.0, 11.0, -16.0, -7.0, 13.0, 11.0, 2.0];
        let r = lpc::autocorrelate(&x7, 6);
        for i in 0..(r.len()) {
            assert_eq!(r[i], y7[i]);
        }
    }

    #[test]
    fn levinson_test() {
        let x7 = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let y4 = vec![1.0, -0.69190537, 0.76150628, -0.34575153];
        let r = lpc::autocorrelate(&x7, 3);
        let (l, _) = lpc::levinson(3, &r);
        for i in 0..(l.len()) {
            assert!((l[i] - y4[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn formant_detection_test() {
        let lpc = [
            1.,
            -1.75325333,
            1.97953403,
            -1.80343314,
            1.20047156,
            0.00740131,
            -0.46918192,
            0.74669944,
            -0.81144139,
            0.5992474,
            -0.22257812,
            0.12155728,
            0.04168977,
        ];
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

        for c in check {
            assert!(c);
        }
    }

    #[test]
    fn peak_detection_test() {
        let lpc = [
            1.,
            -1.75325333,
            1.97953403,
            -1.80343314,
            1.20047156,
            0.00740131,
            -0.46918192,
            0.74669944,
            -0.81144139,
            0.5992474,
            -0.22257812,
            0.12155728,
            0.04168977,
        ];
        let fs = 11025.0f64; // sampling rate
        let peaks = lpc::peak_detection(&lpc, fs);

        const PEAKS_NUM: usize = 11;
        let epsilon = 10.0;
        // let answers: [f64; PEAKS_NUM] = [654.0, 1131.0, 2382.0, 2826.0, 3539.0, -5512.0, -2826.0, -3539.0, -2382.0, -1131.0, -654.0];
        let answers: [f64; PEAKS_NUM] = [
            654.0, 1131.0, 2382.0, 2826.0, 3539.0, 5512.0, 8198.0, 7486.0, 8642.0, 9894.0, 10370.0,
        ];

        let mut check: [bool; PEAKS_NUM] = [false; PEAKS_NUM];
        for peak in peaks {
            let mut check2 = false;
            for i in 0..PEAKS_NUM {
                check[i] = check[i] || ((peak - answers[i]).abs() < epsilon);
                check2 = check2 || (peak - answers[i]).abs() < epsilon;
            }
            assert!(check2);
        }

        for c in check {
            assert!(c);
        }
    }

    /// Helper function to manually downsample the data.
    fn manual_downsample(data: &[f64], factor: usize) -> Vec<f64> {
        data.iter().step_by(factor).cloned().collect()
    }
}
