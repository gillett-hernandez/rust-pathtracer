use crate::hittable::HitRecord;
use crate::math::*;

use std::marker::{Send, Sync};

#[allow(unused_variables)]
pub trait Material: Send + Sync {
    // provide default implementations

    // methods for sampling the bsdf, not related to the light itself

    // evaluate bsdf
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> SingleEnergy {
        SingleEnergy::new(0.0)
    }
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> PDF {
        0.0.into()
    }
    fn generate(&self, hit: &HitRecord, s: Sample2D, wi: Vec3) -> Option<Vec3> {
        None
    }

    // method to sample an emitted light ray with a wavelength and energy
    // can fail when the material is not emissive
    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength, PDF, PDF)> {
        None
    }

    // evaluate the spectral power distribution for the given light and angle
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Option<Vec3>) -> SingleEnergy {
        SingleEnergy::ZERO
    }
    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(&self, hit: &HitRecord, wo: Vec3) -> PDF {
        // hit is passed in to access the UV.
        0.0.into()
    }

    // method to sample the emission spectra at a given uv
    fn sample_emission_spectra(
        &self,
        uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF)> {
        None
    }
}
