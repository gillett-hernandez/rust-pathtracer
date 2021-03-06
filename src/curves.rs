// this module contains a series of SDFs

use crate::math::spectral::{InterpolationMode, Op};
use crate::math::*;

pub use crate::math::spectral::EXTENDED_VISIBLE_RANGE;

pub fn cie_e(power: f32) -> SPD {
    SPD::Linear {
        signal: vec![power],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    }
}

pub fn blackbody(temperature: f32, boost: f32) -> SPD {
    SPD::Blackbody { temperature, boost }
}

pub fn cauchy(a: f32, b: f32) -> SPD {
    SPD::Cauchy { a, b }
}

pub fn red(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![(640.0, 240.0, 240.0, 20.0 * 1.7 * power)],
    }
}

pub fn green(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![(540.0, 240.0, 240.0, 20.0 * 1.7 * power)],
    }
}

pub fn blue(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![(420.0, 240.0, 240.0, 20.0 * 1.7 * power)],
    }
}

pub fn mauve(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![
            (650.0, 300.0, 300.0, power),
            (460.0, 200.0, 400.0, 0.75 * power),
        ],
    }
}

pub fn add_pigment(spd: SPD, wavelength: f32, std_dev1: f32, std_dev2: f32, strength: f32) -> SPD {
    let pigment = SPD::InverseExponential {
        signal: vec![(wavelength, std_dev1, std_dev2, strength)],
    };
    match spd {
        SPD::Machine { seed, mut list } => {
            list.push((Op::Mul, pigment));
            SPD::Machine { seed, list }
        }
        _ => SPD::Machine {
            seed: 1.0,
            list: vec![(Op::Mul, spd), (Op::Mul, pigment)],
        },
    }
}

pub fn void() -> SPD {
    SPD::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    }
}
