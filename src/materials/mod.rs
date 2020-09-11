use crate::math::*;
use crate::TransportMode;

use std::marker::{Send, Sync};

#[allow(unused_variables)]
pub trait Material: Send + Sync {
    // provide default implementations

    // methods for sampling the bsdf, not related to the light itself

    // evaluate bsdf
    fn f(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> SingleEnergy {
        SingleEnergy::new(0.0)
    }
    fn scatter_pdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> PDF {
        0.0.into()
    }
    fn generate(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
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
    fn emission(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Option<Vec3>,
    ) -> SingleEnergy {
        SingleEnergy::ZERO
    }
    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wo: Vec3,
    ) -> PDF {
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

mod diffuse_light;
mod ggx;
mod lambertian;
mod sharp_light;

pub use diffuse_light::DiffuseLight;
pub use ggx::GGX;
pub use lambertian::Lambertian;
pub use sharp_light::SharpLight;

// type required for an id into the Material Table
// pub type MaterialId = u8;
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

#[derive(Clone)]
pub enum MaterialEnum {
    GGX(GGX),
    Lambertian(Lambertian),
    DiffuseLight(DiffuseLight),
    SharpLight(SharpLight),
}

impl From<DiffuseLight> for MaterialEnum {
    fn from(value: DiffuseLight) -> Self {
        MaterialEnum::DiffuseLight(value)
    }
}

impl From<Lambertian> for MaterialEnum {
    fn from(value: Lambertian) -> Self {
        MaterialEnum::Lambertian(value)
    }
}

impl From<SharpLight> for MaterialEnum {
    fn from(value: SharpLight) -> Self {
        MaterialEnum::SharpLight(value)
    }
}

impl From<GGX> for MaterialEnum {
    fn from(value: GGX) -> Self {
        MaterialEnum::GGX(value)
    }
}

impl MaterialEnum {
    pub fn get_name(&self) -> &str {
        match self {
            MaterialEnum::GGX(_inner) => GGX::NAME,
            MaterialEnum::Lambertian(_inner) => Lambertian::NAME,
            MaterialEnum::SharpLight(_inner) => SharpLight::NAME,
            MaterialEnum::DiffuseLight(_inner) => DiffuseLight::NAME,
        }
    }
}

impl Material for MaterialEnum {
    fn scatter_pdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> PDF {
        debug_assert!(lambda > 0.0, "{}", lambda);
        debug_assert!(wi.0.is_finite().all());
        debug_assert!(wo.0.is_finite().all());
        match self {
            MaterialEnum::GGX(inner) => inner.scatter_pdf(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::Lambertian(inner) => {
                inner.scatter_pdf(lambda, uv, transport_mode, wi, wo)
            }
            MaterialEnum::SharpLight(inner) => {
                inner.scatter_pdf(lambda, uv, transport_mode, wi, wo)
            }
            MaterialEnum::DiffuseLight(inner) => {
                inner.scatter_pdf(lambda, uv, transport_mode, wi, wo)
            }
        }
    }
    fn generate(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::GGX(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::Lambertian(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::SharpLight(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::DiffuseLight(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
        }
    }
    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength, PDF, PDF)> {
        match self {
            MaterialEnum::GGX(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
            MaterialEnum::Lambertian(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
            MaterialEnum::SharpLight(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
            MaterialEnum::DiffuseLight(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
        }
    }
    fn f(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> SingleEnergy {
        debug_assert!(lambda > 0.0, "{}", lambda);
        debug_assert!(wi.0.is_finite().all());
        debug_assert!(wo.0.is_finite().all());
        match self {
            MaterialEnum::GGX(inner) => inner.f(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::Lambertian(inner) => inner.f(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::SharpLight(inner) => inner.f(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::DiffuseLight(inner) => inner.f(lambda, uv, transport_mode, wi, wo),
        }
    }
    fn emission(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Option<Vec3>,
    ) -> SingleEnergy {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::GGX(inner) => inner.emission(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::Lambertian(inner) => inner.emission(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::SharpLight(inner) => inner.emission(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::DiffuseLight(inner) => inner.emission(lambda, uv, transport_mode, wi, wo),
        }
    }
    fn sample_emission_spectra(
        &self,
        uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF)> {
        match self {
            MaterialEnum::GGX(inner) => {
                inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample)
            }
            MaterialEnum::Lambertian(inner) => {
                inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample)
            }
            MaterialEnum::SharpLight(inner) => {
                inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample)
            }
            MaterialEnum::DiffuseLight(inner) => {
                inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample)
            }
        }
    }
}

// impl std::convert::Into<usize> for MaterialId {
//     fn into(self) -> usize {
//         usize::from(self)
//     }
// }

// pub struct MaterialTable {
//     pub materials: Vec<Box<dyn Material>>,
// }
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
