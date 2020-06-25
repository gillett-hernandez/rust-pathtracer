use crate::hittable::HitRecord;
use crate::math::*;

use std::marker::{Send, Sync};

pub trait Material: Send + Sync {
    // provide default implementations
    // methods for sampling the bsdf
    fn value(&self, _hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> f32 {
        0.0
    }
    fn generate(&self, _hit: &HitRecord, _s: Sample2D, _wi: Vec3) -> Option<Vec3> {
        None
    }
    // method to sample an emitted light ray with a wavelength and energy
    fn sample_emission(
        &self,
        _point: Point3,
        _normal: Vec3,
        _wavelength_range: Bounds1D,
        _scatter_sample: Sample2D,
        _wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength)> {
        None
    }
    fn sample_emission_spectra(
        &self,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<f32> {
        None
    }
    // evaluate bsdf
    fn f(&self, _hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> SingleEnergy {
        SingleEnergy::ZERO
    }
    // evaluate the spectral power distribution for the given light and angle
    fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
        SingleEnergy::ZERO
    }
}
