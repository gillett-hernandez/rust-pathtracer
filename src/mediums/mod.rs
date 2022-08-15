use crate::prelude::*;

use std::marker::{Send, Sync};

mod hg;
mod rayleigh;

// exporting for use in parsing
pub use hg::HenyeyGreensteinHomogeneous;
pub use rayleigh::Rayleigh;

#[allow(unused_variables)]
pub trait Medium<L: Field, E: Field> {
    // phase function. analogous to the bsdf
    fn p(&self, lambda: L, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> E;
    // sample phase function, once scattering has been determined using `sample`, sample_p allows the exit direction to be determined
    // analogous to sampling the bsdf
    fn sample_p(
        &self,
        lambda: L,
        uvw: (f32, f32, f32),
        wi: Vec3,
        sample: Sample2D,
    ) -> (Vec3, PDF<E, SolidAngle>);
    // evaluate transmittance along two points.
    fn tr(&self, lambda: L, p0: Point3, p1: Point3) -> E;
    // if the medium is not homogeneous, the uvw for many points along the ray would be required and some source for the density would also be required, i.e. an openvdb implementation
    // maybe raymarching would be required for this, need to research more.
    // TODO: implement some Point3 -> UVW trait method on potentially a new trait, such that nonhomogeneous mediums can be implemented
    // sample transmittance, i.e. determine how far a ray reaches through this medium
    // returns the point, the extinction along this path, and whether we scattered off the medium or not.
    fn sample(&self, lambda: L, ray: Ray, s: Sample1D) -> (Point3, E, bool);
    // typically, a medium has a few parameters that need to be taken into account.
    // sigma_s == scattering cross section, a parameter that represents how strongly the material out-scatters light
    // sigma_a == absorbtion cross section, a parameter that represents how strongly the material absorbs light
    // sigma_t == total cross section, a derived parameter that represents how strongly the material extinguishes light by any means
    // sigma_t = sigma_a + sigma_s

    // evaluate emission at some uvw. assuming homogeneity, uvw can be discarded.
    fn emission(&self, lambda: L, wo: Vec3, uvw: (f32, f32, f32)) -> E {
        0.0.into()
    }
    // sample emission to exit the material for use in bdpt. again, assuming homogeneity, uvw can be ignored.
    fn sample_emission(&self, lambda: L, uvw: (f32, f32, f32)) -> (Vec3, E) {
        (Vec3::ZERO, 0.0.into())
    }
}

#[macro_export]
macro_rules! generate_medium_enum {
    ( $name:ident, $l: ty, $e: ty, $( $s:ident),+) => {

        #[derive(Clone)]
        pub enum $name {
            $(
                $s($s),
            )+
        }

        impl Medium<$l, $e> for $name {
            fn p(&self, lambda: $l, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> $e {
                match self {
                    $(
                        MediumEnum::$s(inner) => inner.p(lambda, uvw, wi, wo),
                    )+
                }
            }
            fn sample_p(&self, lambda: $l, uvw: (f32, f32, f32), wi: Vec3, s: Sample2D) -> (Vec3, PDF<$e, SolidAngle>) {
                match self {
                    $(
                        MediumEnum::$s(inner) => inner.sample_p(lambda, uvw, wi, s),
                    )+

                }
            }
            fn sample(&self, lambda: $l, ray: Ray, s: Sample1D) -> (Point3, $e, bool) {
                match self {
                    $(
                        MediumEnum::$s(inner)  => inner.sample(lambda, ray, s),
                    )+
                }
            }
            fn tr(&self, lambda: $l, p0: Point3, p1: Point3) -> $e {
                match self {
                    $(
                        MediumEnum::$s(inner)  => inner.tr(lambda, p0, p1),
                    )+
                }
            }
            fn emission(&self, lambda: $l, wo: Vec3, uvw: (f32, f32, f32)) -> $e {
                match self {
                    $(
                        MediumEnum::$s(inner)  => inner.emission(lambda, wo, uvw),
                    )+
                }
            }
            fn sample_emission(&self, lambda: $l, uvw: (f32, f32, f32)) -> (Vec3, $e) {
                match self {
                    $(
                        MediumEnum::$s(inner)  => inner.sample_emission(lambda, uvw),
                    )+
                }
            }
        }
    }
}

generate_medium_enum! {MediumEnum, f32, f32, HenyeyGreensteinHomogeneous, Rayleigh}

unsafe impl Send for MediumEnum {}
unsafe impl Sync for MediumEnum {}

pub type MediumTable = Vec<MediumEnum>;
