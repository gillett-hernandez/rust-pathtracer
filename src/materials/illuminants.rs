// this module contains a series of SDFs and corresponding materials for use as lights

use super::DiffuseLight;
use crate::math::SDF;
use crate::math::*;

pub fn E() -> SDF {
    SDF::Linear {
        signal: vec![1.0],
        bounds: Bounds1D::new(380.0, 780.0),
    }
}

pub const DiffuseLightE: DiffuseLight = DiffuseLight::new(E());

pub fn void() -> SDF {
    SDF::Linear {
        signal: vec![0.0],
        bounds: Bounds1D::new(380.0, 780.0),
    }
}

pub const DiffuseVoid: DiffuseLight = DiffuseLight::new(void());
