use ndarray::prelude::*;

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

pub fn lpctest() -> Result<(), &'static str> {
    // Example autocorrelation coefficients
    let r: [f64; 5] = [1.0, -0.5, 0.25, 0.125, 0.0625];

    // LPC coefficients
    let mut a = [0.0; 5];

    // Compute LPC coefficients
    levinson(&r, &mut a)?;

    println!("LPC Coefficients: {:?}", a);
    Ok(())
}
