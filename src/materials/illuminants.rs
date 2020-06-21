// this module contains a series of SDFs and corresponding materials for use as lights

use super::DiffuseLight;
use crate::math::SDF;
use crate::math::*;

pub fn cie_e(power: f32) -> SDF {
    SDF::Linear {
        signal: vec![power],
        bounds: Bounds1D::new(370.0, 790.0),
    }
}

pub fn void() -> SDF {
    SDF::Linear {
        signal: vec![0.0],
        bounds: Bounds1D::new(370.0, 790.0),
    }
}
