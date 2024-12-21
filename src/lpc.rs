use ndarray::prelude::*;
use rustfft::{num_complex::{Complex, ComplexFloat}, FftPlanner};

fn levinson(
    r: &[f64],     // Autocorrelation coefficients
    a: &mut [f64], // LPC coefficients
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
fn autocorrelation_frequency_domain(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let n = signal.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    // Prepare input: real signal to complex
    let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut buffer2: Vec<f64> = vec![0.0; n];

    // Perform FFT
    fft.process(&mut buffer);

    // Compute power spectrum (magnitude squared)
    for i in 0..n {
         buffer2[i] = buffer[i].norm_sqr();
    }

    // Inverse FFT
    let ifft = FftPlanner::<f64>::new().plan_fft_inverse(n);
    //let mut autocorr_complex = vec![Complex::new(0.0, 0.0); n];
    ifft.process(&mut buffer);

    // Extract real part and normalize
    let autocorr: Vec<f64> = buffer.iter().map(|c| c.re / n as f64).collect();

    // Return autocorrelation up to max_lag
    autocorr[..=max_lag].to_vec()
}

pub fn lpctest() {
    // Test signal (a simple sine wave or random signal)
    let signal = vec![1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, -0.05, -0.1, -0.2];

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
