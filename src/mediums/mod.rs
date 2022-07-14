use crate::math::*;

use std::marker::{Send, Sync};

mod hg;
mod rayleigh;

// exporting for use in parsing
pub use hg::HenyeyGreensteinHomogeneous;
pub use rayleigh::Rayleigh;

pub trait Medium {
    // phase function. analogous to the bsdf
    fn p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32;
    // sample phase function, once scattering has been determined using `sample`, sample_p allows the exit direction to be determined
    // analogous to sampling the bsdf
    fn sample_p(
        &self,
        lambda: f32,
        uvw: (f32, f32, f32),
        wi: Vec3,
        sample: Sample2D,
    ) -> (Vec3, f32);
    // sample transmittance, i.e. determine how far a ray reaches through this medium
    // if the medium is not homogeneous, the uvw for each point along the ray would be required and some source for the density would also be required, i.e. an openvdb implementation
    // TODO: implement some Point3 -> UVW trait method on potentially a new trait, such that nonhomogeneous mediums can be implemented
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool);
    // evaluate transmittance along two points.
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32;
    // evaluate emission at some uvw. assuming homogeneity, uvw can be discarded.
    fn emission(&self, _lambda: f32, _wo: Vec3, _uvw: (f32, f32, f32)) -> SingleEnergy {
        0.0.into()
    }
    // sample emission to exit the material for use in bdpt. again, assuming homogeneity, uvw can be ignored.
    fn sample_emission(&self, _lambda: f32, _uvw: (f32, f32, f32)) -> (Vec3, SingleEnergy) {
        (Vec3::ZERO, 0.0.into())
    }
}

#[derive(Clone)]
pub enum MediumEnum {
    HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous),
    Rayleigh(Rayleigh),
}

impl Medium for MediumEnum {
    fn p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.p(lambda, uvw, wi, wo),
            MediumEnum::Rayleigh(inner) => inner.p(lambda, uvw, wi, wo),
        }
    }
    fn sample_p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, s: Sample2D) -> (Vec3, f32) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample_p(lambda, uvw, wi, s),
            MediumEnum::Rayleigh(inner) => inner.sample_p(lambda, uvw, wi, s),
        }
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample(lambda, ray, s),
            MediumEnum::Rayleigh(inner) => inner.sample(lambda, ray, s),
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.tr(lambda, p0, p1),
            MediumEnum::Rayleigh(inner) => inner.tr(lambda, p0, p1),
        }
    }
    fn emission(&self, lambda: f32, wo: Vec3, uvw: (f32, f32, f32)) -> SingleEnergy {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.emission(lambda, wo, uvw),
            MediumEnum::Rayleigh(inner) => inner.emission(lambda, wo, uvw),
        }
    }
    fn sample_emission(&self, lambda: f32, uvw: (f32, f32, f32)) -> (Vec3, SingleEnergy) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample_emission(lambda, uvw),
            MediumEnum::Rayleigh(inner) => inner.sample_emission(lambda, uvw),
        }
    }
}

unsafe impl Send for MediumEnum {}
unsafe impl Sync for MediumEnum {}

pub type MediumTable = Vec<MediumEnum>;
