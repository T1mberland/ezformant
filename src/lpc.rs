use ndarray::prelude::*;
use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};

fn levinson(
    r: &[f32],     // Autocorrelation coefficients
    a: &mut [f32], // LPC coefficients
) -> Result<(), &'static str> {
    let m = r.len() - 1; // Order of LPC
    if r.is_empty() || a.len() != m + 1 {
        return Err("Invalid input sizes");
    }

    let mut e = r[0]; // Prediction error (initially R[0])

    if e == 0.0 {
        return Err("Autocorrelation R[0] cannot be zero");
    }

    a[0] = 1.0; // First LPC coefficient (A[0] = 1.0)

    for i in 1..=m {
        // Compute the reflection coefficient (K)
        let mut k = -r[i];
        for j in 1..i {
            k -= a[j] * r[i - j];
        }
        k /= e;

        // Update LPC coefficients
        for j in (1..i).rev() {
            a[j] += k * a[i - j];
        }
        a[i] = k;

        // Update prediction error
        e *= 1.0 - k * k;
        if e <= 0.0 {
            return Err("Prediction error became non-positive, indicating instability");
        }
    }

    Ok(())
}

/// Computes autocorrelation using the frequency-domain method.
fn autocorrelation_frequency_domain(signal: &[f32], max_lag: usize) -> Vec<f32> {
    let n = signal.len();
    if max_lag >= n {
        panic!("max_lag must be less than the length of the signal");
    }

    // Set up FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert real input to complex
    let mut buffer: Vec<Complex<f32>> = signal
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
    let scale = 1.0 / n as f32;
    let autocorr_time_domain: Vec<f32> = buffer
        .iter()
        .map(|c| c.re * scale)
        .collect();

    // Return up to max_lag
    autocorr_time_domain[..=max_lag].to_vec()
}

/// Compute the frequency response of the LPC filter
fn compute_frequency_response(lpc_coeffs: &[f32], sample_rate: f32, num_points: usize) -> Vec<(f32, f32)> {
    let mut response = Vec::new();
    for i in 0..num_points {
        let freq = i as f32 / num_points as f32 * sample_rate / 2.0; // Frequency in Hz
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let z = Complex::new(omega.cos(), -omega.sin()); // Complex exponential
        let denominator: Complex<f32> = lpc_coeffs.iter().enumerate().map(|(k, &a_k)| z.powi(-(k as i32)) * a_k).sum();
        let h = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) + denominator); // Transfer function
        response.push((freq, h.norm()));
    }
    response
}

pub fn lpctest() {
    // Test signal (a simple sine wave or random signal)
    let signal = vec![1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, -0.05, -0.1, -0.2, 1.0, 0.0, -1.0, 0.03];

    // Calculate autocorrelation
    let max_lag = 4;
    let autocorr = autocorrelation_frequency_domain(&signal, max_lag);

    println!("Autocorrelation: {:?}", autocorr);

    // Prepare LPC coefficients vector
    let order = max_lag; // LPC order
    let mut lpc_coeffs = vec![0.0; order + 1];

    // Call the Levinson-Durbin function
    match levinson(&autocorr, &mut lpc_coeffs) {
        Ok(()) => {
            println!("LPC Coefficients: {:?}", lpc_coeffs);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
