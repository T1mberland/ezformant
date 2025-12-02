pub fn difference_function(signal: &[f64], t: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..signal.len() {
        if i + t >= signal.len() {
            break;
        }
        acc += (signal[i] - signal[i + t]).powi(2);
    }

    return acc;
}

pub fn cmnd_first_peak(signal: &[f64], t_max: usize, threshold: f64) -> Option<usize> {
    let mut d_sum = 0.0;
    for t in 1..t_max {
        let d = difference_function(signal, t);
        d_sum += d;

        let cmnd_val = d * (t as f64) / d_sum;

        if cmnd_val < threshold {
            return Some(t);
        }
    }

    return None;
}

pub fn pitch_detection_yin(signal: &[f64], sampling_rate: f64) -> f64 {
    return match cmnd_first_peak(signal, signal.len() / 2, 0.1) {
        None => -1.0,
        Some(k) => sampling_rate / (k as f64),
    };
}
