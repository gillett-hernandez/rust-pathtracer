use crate::math::*;
use crate::world::TransportMode;

use std::marker::{Send, Sync};

#[allow(unused_variables)]
pub trait Material: Send + Sync {
    // provide default implementations

    // methods for sampling the bsdf, not related to the light itself

    // evaluate bsdf
    fn bsdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        (SingleEnergy::new(0.0), 0.0.into())
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

    fn is_medium(&self) -> bool {
        false
    }
    fn eval_phase(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (SingleEnergy, PDF) {
        (0.0.into(), 0.0.into())
    }
    fn sample_phase(&self, lambda: f32, wi: Vec3, s: Sample2D) -> (Vec3, PDF) {
        (Vec3::ZERO, 0.0.into())
    }
    fn tr(&self, lambda: f32, distance: f32) -> SingleEnergy {
        // Tr
        0.0.into()
    }
    fn sample_tr(
        &self,
        lambda: f32,
        time_bounds: Bounds1D,
        s: Sample1D,
    ) -> (f32, SingleEnergy, PDF) {
        if s.x < 0.5 {
            (1.0, self.tr(lambda, 1.0), 0.1.into())
        } else {
            let t = time_bounds.lerp(0.5);
            (
                t,                                 // time
                self.tr(lambda, t),                // Tr
                (0.9 / time_bounds.span()).into(), //PDF of scattering within time bounds
            )
        }
    }
}

mod diffuse_light;
mod ggx;
mod lambertian;
mod passthrough;
mod sharp_light;
mod volumes;

pub use diffuse_light::DiffuseLight;
pub use ggx::{reflect, refract, GGX};
pub use lambertian::Lambertian;
pub use passthrough::PassthroughFilter;
pub use sharp_light::SharpLight;
pub use volumes::*;

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
    PassthroughFilter(PassthroughFilter),
    DiffuseLight(DiffuseLight),
    SharpLight(SharpLight),
    Isotropic(Isotropic),
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
impl From<Isotropic> for MaterialEnum {
    fn from(value: Isotropic) -> Self {
        MaterialEnum::Isotropic(value)
    }
}

impl From<PassthroughFilter> for MaterialEnum {
    fn from(value: PassthroughFilter) -> Self {
        MaterialEnum::PassthroughFilter(value)
    }
}

impl MaterialEnum {
    pub fn get_name(&self) -> &str {
        match self {
            MaterialEnum::GGX(_inner) => GGX::NAME,
            MaterialEnum::PassthroughFilter(_inner) => PassthroughFilter::NAME,
            MaterialEnum::Lambertian(_inner) => Lambertian::NAME,
            MaterialEnum::SharpLight(_inner) => SharpLight::NAME,
            MaterialEnum::DiffuseLight(_inner) => DiffuseLight::NAME,
            MaterialEnum::Isotropic(_inner) => Isotropic::NAME,
        }
    }
}

impl Material for MaterialEnum {
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
            MaterialEnum::PassthroughFilter(inner) => {
                inner.generate(lambda, uv, transport_mode, s, wi)
            }
            MaterialEnum::Lambertian(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::SharpLight(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::DiffuseLight(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
            MaterialEnum::Isotropic(inner) => inner.generate(lambda, uv, transport_mode, s, wi),
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
            MaterialEnum::Isotropic(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
            MaterialEnum::PassthroughFilter(inner) => inner.sample_emission(
                point,
                normal,
                wavelength_range,
                scatter_sample,
                wavelength_sample,
            ),
        }
    }
    fn bsdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        debug_assert!(lambda > 0.0, "{}", lambda);
        debug_assert!(wi.0.is_finite().all());
        debug_assert!(wo.0.is_finite().all());
        debug_assert!(wo != Vec3::ZERO);
        match self {
            MaterialEnum::GGX(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::PassthroughFilter(inner) => {
                inner.bsdf(lambda, uv, transport_mode, wi, wo)
            }
            MaterialEnum::Lambertian(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::SharpLight(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::DiffuseLight(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),
            MaterialEnum::Isotropic(inner) => inner.bsdf(lambda, uv, transport_mode, wi, wo),
        }
    }
    fn emission(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
    ) -> SingleEnergy {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::GGX(inner) => inner.emission(lambda, uv, transport_mode, wi),
            MaterialEnum::PassthroughFilter(inner) => {
                inner.emission(lambda, uv, transport_mode, wi)
            }
            MaterialEnum::Lambertian(inner) => inner.emission(lambda, uv, transport_mode, wi),
            MaterialEnum::SharpLight(inner) => inner.emission(lambda, uv, transport_mode, wi),
            MaterialEnum::DiffuseLight(inner) => inner.emission(lambda, uv, transport_mode, wi),
            MaterialEnum::Isotropic(inner) => inner.emission(lambda, uv, transport_mode, wi),
        }
    }
    fn outer_medium_id(&self, uv: (f32, f32)) -> usize {
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.outer_medium_id(uv),
            MaterialEnum::Isotropic(inner) => inner.outer_medium_id(uv),

            MaterialEnum::GGX(inner) => inner.outer_medium_id(uv),
            MaterialEnum::Lambertian(inner) => inner.outer_medium_id(uv),
            MaterialEnum::SharpLight(inner) => inner.outer_medium_id(uv),
            MaterialEnum::DiffuseLight(inner) => inner.outer_medium_id(uv),
        }
    }
    fn inner_medium_id(&self, uv: (f32, f32)) -> usize {
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.inner_medium_id(uv),
            MaterialEnum::Isotropic(inner) => inner.inner_medium_id(uv),

            MaterialEnum::GGX(inner) => inner.inner_medium_id(uv),
            MaterialEnum::Lambertian(inner) => inner.inner_medium_id(uv),
            MaterialEnum::SharpLight(inner) => inner.inner_medium_id(uv),
            MaterialEnum::DiffuseLight(inner) => inner.inner_medium_id(uv),
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
            MaterialEnum::PassthroughFilter(inner) => {
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
            MaterialEnum::Isotropic(inner) => {
                inner.sample_emission_spectra(uv, wavelength_range, wavelength_sample)
            }
        }
    }
    fn sample_tr(
        &self,
        lambda: f32,
        time_bounds: Bounds1D,
        s: Sample1D,
    ) -> (f32, SingleEnergy, PDF) {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.sample_tr(lambda, time_bounds, s),
            MaterialEnum::Isotropic(inner) => inner.sample_tr(lambda, time_bounds, s),

            MaterialEnum::GGX(inner) => inner.sample_tr(lambda, time_bounds, s),
            MaterialEnum::Lambertian(inner) => inner.sample_tr(lambda, time_bounds, s),
            MaterialEnum::SharpLight(inner) => inner.sample_tr(lambda, time_bounds, s),
            MaterialEnum::DiffuseLight(inner) => inner.sample_tr(lambda, time_bounds, s),
        }
    }
    fn sample_phase(&self, lambda: f32, wi: Vec3, s: Sample2D) -> (Vec3, PDF) {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.sample_phase(lambda, wi, s),
            MaterialEnum::Isotropic(inner) => inner.sample_phase(lambda, wi, s),

            MaterialEnum::GGX(inner) => inner.sample_phase(lambda, wi, s),
            MaterialEnum::Lambertian(inner) => inner.sample_phase(lambda, wi, s),
            MaterialEnum::SharpLight(inner) => inner.sample_phase(lambda, wi, s),
            MaterialEnum::DiffuseLight(inner) => inner.sample_phase(lambda, wi, s),
        }
    }
    fn eval_phase(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (SingleEnergy, PDF) {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.eval_phase(lambda, wi, wo),
            MaterialEnum::Isotropic(inner) => inner.eval_phase(lambda, wi, wo),

            MaterialEnum::GGX(inner) => inner.eval_phase(lambda, wi, wo),
            MaterialEnum::Lambertian(inner) => inner.eval_phase(lambda, wi, wo),
            MaterialEnum::SharpLight(inner) => inner.eval_phase(lambda, wi, wo),
            MaterialEnum::DiffuseLight(inner) => inner.eval_phase(lambda, wi, wo),
        }
    }
    fn tr(&self, lambda: f32, distance: f32) -> SingleEnergy {
        debug_assert!(lambda > 0.0, "{}", lambda);
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.tr(lambda, distance),
            MaterialEnum::Isotropic(inner) => inner.tr(lambda, distance),

            MaterialEnum::GGX(inner) => inner.tr(lambda, distance),
            MaterialEnum::Lambertian(inner) => inner.tr(lambda, distance),
            MaterialEnum::SharpLight(inner) => inner.tr(lambda, distance),
            MaterialEnum::DiffuseLight(inner) => inner.tr(lambda, distance),
        }
    }
    fn is_medium(&self) -> bool {
        match self {
            MaterialEnum::PassthroughFilter(inner) => inner.is_medium(),
            MaterialEnum::Isotropic(inner) => inner.is_medium(),

            MaterialEnum::GGX(inner) => inner.is_medium(),
            MaterialEnum::Lambertian(inner) => inner.is_medium(),
            MaterialEnum::SharpLight(inner) => inner.is_medium(),
            MaterialEnum::DiffuseLight(inner) => inner.is_medium(),
        }
    }
}
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
