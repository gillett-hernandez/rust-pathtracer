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

const HCC2: f32 = 1.1910429723971884140794892e-29;
const HKC: f32 = 1.438777085924334052222404423195819240925e-2;

pub fn blackbody(temperature: f32, lambda: f32) -> f32 {
    let lambda = lambda * 1e-9;

    lambda.powi(-5) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

pub fn max_blackbody_lambda(temp: f32) -> f32 {
    2.8977721e-3 / (temp * 1e-9)
}
