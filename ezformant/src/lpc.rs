use aberth::AberthSolver;
use rustfft::num_complex::{Complex, ComplexFloat};

/// Applies a pre-emphasis filter to a signal in-place.
///
/// # Arguments
///
/// * `signal` - The signal to be filtered.
/// * `alpha`  - The pre-emphasis coefficient.
///
/// # Example
///
/// ```ignore
/// let mut samples = vec![1.0, 2.0, 3.0];
/// pre_emphasis(&mut samples, 0.95);
/// ```
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
/// A vector containing autocorrelation values from lag 0 to `maxlag`.
pub fn autocorrelate(signal: &[f64], maxlag: usize) -> Vec<f64> {
    let n = signal.len();
    let mut result = Vec::with_capacity(maxlag + 1);

    for lag in 0..=maxlag {
        // Only sum while (i + lag) is within the signal’s length
        let r: f64 = signal
            .iter()
            .enumerate()
            .take_while(|(i, _)| i + lag < n)
            .map(|(i, &val)| val * signal[i + lag])
            .sum();

        result.push(r);
    }

    result
}

/// Implements the Levinson-Durbin recursion algorithm iteratively.
///
/// # Arguments
///
/// * `order`  - The order of the recursion (filter).
/// * `r`      - A slice of f64 representing the autocorrelation coefficients.
///              Must have length >= `order + 1`.
///
/// # Returns
///
/// A tuple containing:
/// - A vector of filter coefficients `[a0, a1, ..., a_order]` (with `a0 = 1.0`).
/// - The final prediction error (`E`).
pub fn levinson(order: usize, r: &[f64]) -> (Vec<f64>, f64) {
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;

    let mut e = r[0];

    for i in 1..=order {
        // Compute the reflection coefficient lambda
        let mut lambda = 0.0;
        for j in 0..i {
            lambda += a[j] * r[i - j];
        }
        lambda = -lambda / e;

        // Update the coefficients a[0..=i]
        for j in 0..=(i / 2) {
            let temp = a[j] + lambda * a[i - j];
            a[i - j] += lambda * a[j];
            a[j] = temp;
        }

        // Update the prediction error
        e *= 1.0 - lambda * lambda;
    }

    (a, e)
}

/// Computes the frequency response of the LPC filter.
///
/// # Arguments
///
/// * `lpc_coeffs` - The LPC coefficients.
/// * `sample_rate` - The sampling rate of the original signal.
/// * `num_points` - The number of frequency points in the response.
///
/// # Returns
///
/// A vector of `(frequency, magnitude)` pairs.
pub fn compute_frequency_response(
    lpc_coeffs: &[f64],
    sample_rate: f64,
    num_points: usize,
) -> Vec<(f64, f64)> {
    let mut response = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let freq = i as f64 / num_points as f64 * sample_rate / 2.0;
        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
        let z = Complex::new(omega.cos(), -omega.sin()); // e^{-j omega}

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

/// Detects peaks (roots' angles) given LPC coefficients using the Aberth method.
///
/// # Arguments
///
/// * `lpc_coeffs` - The LPC coefficients.
/// * `sample_rate` - The sampling rate of the original signal.
///
/// # Returns
///
/// A vector of frequencies (in Hz) corresponding to the angles of the polynomial roots.
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
        // Convert angle to a frequency in Hz
        if theta >= 0.0 {
            peaks.push(theta * sample_rate / (2.0 * std::f64::consts::PI));
        } else if theta >= -std::f64::consts::PI && theta < 0.0 {
            // Shift negative angle into [0, 2π)
            let shifted = theta + 2.0 * std::f64::consts::PI;
            peaks.push(shifted * sample_rate / (2.0 * std::f64::consts::PI));
        }
    }

    peaks
}

/// Performs formant detection from LPC coefficients by selecting valid peaks.
///
/// # Arguments
///
/// * `lpc_coeffs`  - The LPC coefficients.
/// * `sample_rate` - The sampling rate of the original signal.
///
/// # Returns
///
/// A vector of formant frequencies in Hz.
pub fn formant_detection(lpc_coeffs: &[f64], sample_rate: f64) -> Vec<f64> {
    let mut peaks = peak_detection(lpc_coeffs, sample_rate);
    let mut formants = Vec::with_capacity(peaks.len());

    // Filter out frequencies near 0 Hz or beyond Nyquist
    let low_cutoff = 10.0;
    let high_cutoff = (sample_rate / 2.0) - 10.0;

    for peak in peaks.drain(..) {
        if peak > low_cutoff && peak < high_cutoff {
            formants.push(peak);
        }
    }

    formants.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    formants
}

/* ------------------------------------------ */
/* ------------------- Tests ---------------- */
/* ------------------------------------------ */
// Uncomment to enable unit tests
/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_emphasis() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0];
        pre_emphasis(&mut samples, 0.95);
        // Write your assertions here
    }
}
*/
