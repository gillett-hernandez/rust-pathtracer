use crate::math::{Sample2D, Sample3D, Vec3, PI};

pub fn power_heuristic(a: f32, b: f32) -> f32 {
    (a * a) / (a * a + b * b)
}

pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn w(x: f32, mul: f32, offset: f32, sigma: f32) -> f32 {
    mul * (-(x - offset).powi(2) / sigma).exp() / (sigma * PI).sqrt()
}
