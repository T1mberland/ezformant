use ndarray::prelude::*;

// p = lpc_size

fn matrix_a_generator(signal: &[f32], n:usize, i: usize, j: usize) -> f32 {
    if n+i<j+1 {
        0.0f32
    } else if n+i-(j+1) >= signal.len() {
        0.0f32
    } else {
        signal[(n+i) - (j+1)]
    }
}

fn calc_matrix_a(signal: &[f32], n:usize, p: usize) -> Array2<f32> {
    let sample_size = signal.len();
    Array::from_shape_fn((sample_size, p), |(i,j)| matrix_a_generator(signal, n, i, j))
}

fn calc_moore_penrose_inverse(signal: &[f32], n:usize, p: usize) {
    let a = calc_matrix_a(signal, n, p);
    let at = Array::t(&a);
    let ata = at.dot(&a);
}

