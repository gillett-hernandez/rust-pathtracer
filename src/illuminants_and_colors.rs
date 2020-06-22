// this module contains a series of SDFs

use crate::math::spectral::Op;
use crate::math::*;

const VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);

pub fn cie_e(power: f32) -> SPD {
    SPD::Linear {
        signal: vec![power],
        bounds: VISIBLE_RANGE,
    }
}

pub fn blackbody(temperature: f32, boost: f32) -> SPD {
    SPD::Blackbody { temperature, boost }
}

pub fn red(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![
            (650.0, 2400.0, 50.0 * 1.7 * power),
        ],
    }
}

pub fn green(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![(540.0, 2400.0, 50.0 * 1.7 * power)],
    }
}

pub fn blue(power: f32) -> SPD {
    SPD::Exponential {
        signal: vec![(400.0, 2400.0, 50.0 * 1.7 * power)],
    }
}

pub fn add_pigment(mut spd: SPD, wavelength: f32, std_dev: f32, strength: f32) -> SPD {
    let pigment = SPD::InverseExponential {
        signal: vec![(wavelength, std_dev, strength)],
    };
    match (spd) {
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
        bounds: VISIBLE_RANGE,
    }
}
