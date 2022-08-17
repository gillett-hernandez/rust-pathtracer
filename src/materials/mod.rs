use crate::prelude::*;

use std::marker::{Send, Sync};

mod diffuse_light;
mod ggx;
mod lambertian;
mod passthrough;
mod sharp_light;

pub use diffuse_light::DiffuseLight;
pub use ggx::{reflect, refract, GGX};
pub use lambertian::Lambertian;
pub use passthrough::PassthroughFilter;
pub use sharp_light::SharpLight;

#[allow(unused_variables)]
pub trait Material<L: Field, E: Field>: Send + Sync {
    // provide default implementations

    // methods for sampling the bsdf, not related to the light itself

    // evaluate bsdf
    fn bsdf(
        &self,
        lambda: L,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (E, PDF<E, SolidAngle>);
    fn generate_and_evaluate(
        &self,
        lambda: L,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (E, Option<Vec3>, PDF<E, SolidAngle>);
    fn generate(
        &self,
        lambda: L,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        self.generate_and_evaluate(lambda, uv, transport_mode, s, wi)
            .1
    }

    fn outer_medium_id(&self, uv: (f32, f32)) -> usize {
        0
    }
    fn inner_medium_id(&self, uv: (f32, f32)) -> usize {
        0
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
    ) -> Option<(
        Ray,
        SingleWavelength,
        PDF<f32, SolidAngle>,
        PDF<f32, Uniform01>,
    )> {
        None
    }

    // evaluate the spectral power distribution for the given light and angle
    fn emission(&self, lambda: L, uv: (f32, f32), transport_mode: TransportMode, wi: Vec3) -> E {
        E::ZERO
    }
    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(
        &self,
        lambda: L,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wo: Vec3,
    ) -> PDF<E, SolidAngle> {
        // hit is passed in to access the UV.
        PDF::new(E::ZERO)
    }

    // method to sample the emission spectra at a given uv
    fn sample_emission_spectra(
        &self,
        uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(L, PDF<E, Uniform01>)> {
        None
    }
}

// type required for an id into the Material Table

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum MaterialId {
    Material(u16),
    Light(u16),
    Camera(u16),
}

impl Default for MaterialId {
    fn default() -> Self {
        MaterialId::Material(0)
    }
}

impl From<u16> for MaterialId {
    fn from(value: u16) -> Self {
        MaterialId::Material(value)
    }
}

impl From<MaterialId> for usize {
    fn from(value: MaterialId) -> Self {
        match value {
            MaterialId::Light(v) => v as usize,
            MaterialId::Camera(v) => v as usize,
            MaterialId::Material(v) => v as usize,
        }
    }
}

#[macro_export]
macro_rules! generate_enum {

    ( $name:ident, ($l: ty, $e: ty), $( $s:ident),+) => {


        impl Material<$l, $e> for $name {
            fn generate(&self,
                lambda: $l,
                uv: (f32, f32),
                transport_mode: TransportMode,
                s: Sample2D,
                wi: Vec3
            ) -> Option<Vec3> {
                match self {
                    $($name::$s(inner) => inner.generate(lambda, uv, transport_mode, s, wi),)+
                }
            }
            fn generate_and_evaluate(&self,
                lambda: $l,
                uv: (f32, f32),
                transport_mode: TransportMode,
                s: Sample2D,
                wi: Vec3,
            ) -> ($e, Option<Vec3>, PDF<$e, SolidAngle>) {
                match self {
                    $($name::$s(inner) => inner.generate_and_evaluate(lambda, uv, transport_mode, s, wi),)+
                }
            }

            // TODO: change this function definition to take a uv and return a Vec3 and normal, or something
            // that way you can have a uv dependence for emission, i.e. a textured light
            fn sample_emission( &self,
                point: Point3,
                normal: Vec3,
                wavelength_range: Bounds1D,
                scatter_sample: Sample2D,
                wavelength_sample: Sample1D,
            ) -> Option<(Ray, WavelengthEnergy<$l, $e>, PDF<$e, SolidAngle>, PDF<$e, Uniform01>)>  {
                match self {
                    $($name::$s(inner) => inner.sample_emission(point, normal, wavelength_range, scatter_sample, wavelength_sample),)+
                }
            }

            fn bsdf(
                &self,
                lambda: $l,
                uv: (f32, f32),
                transport_mode: TransportMode,
                wi: Vec3,
                wo: Vec3,
            ) -> ($e, PDF<$e, SolidAngle>) {
                debug_assert!(lambda > 0.0, "{}", lambda);
                debug_assert!(wi.0.is_finite().all());
                debug_assert!(wo.0.is_finite().all());
                debug_assert!(wo != Vec3::ZERO);
                match self {
                    $($name::$s(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),)+
                }
            }
            fn emission(
                &self,
                lambda: $l,
                uv: (f32, f32),
                transport_mode: TransportMode,
                wi: Vec3,
            ) -> $e {
                debug_assert!(lambda > 0.0, "{}", lambda);
                match self {
                    $($name::$s(inner) => inner.emission(lambda, uv, transport_mode, wi),)+
                }
            }

            fn outer_medium_id(&self, uv: (f32, f32)) -> usize {
                match self {
                    $($name::$s(inner) => inner.outer_medium_id(uv),)+
                }
            }

            fn inner_medium_id(&self, uv: (f32, f32)) -> usize {
                match self {
                    $($name::$s(inner) => inner.inner_medium_id(uv),)+
                }
            }
            fn sample_emission_spectra(
                &self,
                uv: (f32, f32),
                wavelength_range: Bounds1D,
                wavelength_sample: Sample1D,
            ) -> Option<($l, PDF<$e, Uniform01>)> {
                match self {
                    $(
                        $name::$s(inner) => inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample),
                    )+
                }
            }
        }
    };
    ( $name:ident, $( $s:ident),+) => {
        #[derive(Clone)]
        pub enum $name {
            $(
                $s($s),
            )+
        }
        $(
            impl From<$s> for $name {
                fn from(value: $s) -> Self {
                    $name::$s(value)
                }
            }
        )+

        impl $name {
            pub fn get_name(&self) -> &str {
                match self {
                    $($name::$s(_) => $s::NAME,)+
                }
            }
        }
    };
}

generate_enum!(
    MaterialEnum,
    GGX,
    Lambertian,
    DiffuseLight,
    SharpLight,
    PassthroughFilter
);

generate_enum!(
    MaterialEnum,
    (f32, f32),
    GGX,
    Lambertian,
    DiffuseLight,
    SharpLight,
    PassthroughFilter
);

// avoid implementing f32x4 for now, since it needs to be implemented for each constituent Material
// generate_enum!(
//     MaterialEnum,
//     (f32x4, f32x4),
//     GGX,
//     Lambertian,
//     DiffuseLight,
//     SharpLight,
//     PassthroughFilter
// );

pub type MaterialTable = Vec<MaterialEnum>;

// #[cfg(test)]
// mod tests {
// use super::*;

// #[test]
// fn test_lambertian() {
//     let lambertian = Lambertian::new(RGBColor::new(0.9, 0.2, 0.9));
//     // simulate incoming ray from directly above
//     let incoming: Ray = Ray::new(Point3::new(0.0, 0.0, 10.0), -Vec3::Z);
//     let hit = HitRecord::new(0.0, Point3::ZERO, 0.0, Vec3::Z, Some(0), 0);
//     let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
//     let v = lambertian.generate(&hit, &mut sampler, incoming.direction);
//     assert!(v.is_some());
//     let V = v.unwrap();
//     println!("{:?}", lambertian.f(&hit, incoming.direction, V));
//     assert!(lambertian.value(&hit, incoming.direction, V) > 0.0);
//     println!("{:?}", V);
//     assert!(V * Vec3::Z > 0.0);
// }
// #[test]
// fn test_lambertian_integral() {
//     let lambertian = Lambertian::new(RGBColor::new(1.0, 1.0, 1.0));
//     // simulate incoming ray from directly above
//     let incoming: Ray = Ray::new(Point3::new(0.0, 0.0, 10.0), -Vec3::Z);
//     let hit = HitRecord::new(0.0, Point3::ZERO, 0.0, Vec3::Z, Some(0), 0);
//     let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
//     let mut pdf_sum = 0.0;
//     let mut color_sum = RGBColor::ZERO;
//     let N = 1000000;
//     for i in 0..N {
//         let v = lambertian.generate(&hit, &mut sampler, -incoming.direction);
//         assert!(v.is_some());
//         let reflectance = lambertian.f(&hit, -incoming.direction, v.unwrap());
//         let pdf = lambertian.value(&hit, -incoming.direction, v.unwrap());
//         assert!(pdf > 0.0);
//         assert!(v.unwrap() * Vec3::Z > 0.0);
//         pdf_sum += pdf;
//         color_sum += reflectance;
//     }
//     println!("{:?}", color_sum / N as f32);
//     println!("{:?}", pdf_sum / N as f32);
//     assert!((pdf_sum / N as f32 - 1.0).abs() < 10000.0 / N as f32);
// }
// }
