// this module contains a series of SDFs

use crate::math::SDF;
use crate::math::*;

const VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);

pub fn cie_e(power: f32) -> SDF {
    SDF::Linear {
        signal: vec![power],
        bounds: VISIBLE_RANGE,
    }
}

pub fn red(power: f32) -> SDF {
    SDF::Exponential {
        signal: vec![
            (700.0, 500.0, 50.0 * 1.7 * power),
            (550.0, 1000.0, 1.4 * power),
        ],
    }
}

pub fn void() -> SDF {
    SDF::Linear {
        signal: vec![0.0],
        bounds: VISIBLE_RANGE,
    }
}
