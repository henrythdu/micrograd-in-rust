use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

pub fn make_moons(n_samples: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    // 2026 Syntax: use rand::rng() instead of thread_rng()
    let mut rng = rand::rng();

    // Normal::new(mean, std_dev)
    let normal = Normal::new(0.0, noise).expect("Invalid distribution parameters");

    let n_outer = n_samples / 2;
    let n_inner = n_samples - n_outer;

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    // Generate Outer Moon
    for i in 0..n_outer {
        let theta = (i as f64 * PI) / (n_outer as f64);
        let x_pos = theta.cos() + normal.sample(&mut rng);
        let y_pos = theta.sin() + normal.sample(&mut rng);
        x.push(vec![x_pos, y_pos]);
        y.push(1.0);
    }

    // Generate Inner Moon
    for i in 0..n_inner {
        let theta = (i as f64 * PI) / (n_inner as f64);
        // Offset the second moon to create the interleaving effect
        let x_pos = 1.0 - theta.cos() + normal.sample(&mut rng);
        let y_pos = 0.5 - theta.sin() + normal.sample(&mut rng);
        x.push(vec![x_pos, y_pos]);
        y.push(-1.0);
    }

    (x, y)
}
