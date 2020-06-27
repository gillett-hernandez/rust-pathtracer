use crate::hittable::*;
use crate::math::*;

pub use crate::material::Material;

mod diffuse_light;
mod ggx;
mod lambertian;
mod parallel_light;

pub use diffuse_light::DiffuseLight;
pub use ggx::GGX;
pub use lambertian::Lambertian;
pub use parallel_light::ParallelLight;

// type required for an id into the Material Table
// pub type MaterialId = u8;
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum MaterialId {
    Material(u8),
    Light(u16),
    Camera(u8),
}

impl From<u8> for MaterialId {
    fn from(value: u8) -> Self {
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
    ParallelLight(ParallelLight),
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

impl From<ParallelLight> for MaterialEnum {
    fn from(value: ParallelLight) -> Self {
        MaterialEnum::ParallelLight(value)
    }
}

impl From<GGX> for MaterialEnum {
    fn from(value: GGX) -> Self {
        MaterialEnum::GGX(value)
    }
}

impl Material for MaterialEnum {
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> PDF {
        match self {
            MaterialEnum::GGX(inner) => inner.value(hit, wi, wo),
            MaterialEnum::Lambertian(inner) => inner.value(hit, wi, wo),
            MaterialEnum::ParallelLight(inner) => inner.value(hit, wi, wo),
            MaterialEnum::DiffuseLight(inner) => inner.value(hit, wi, wo),
        }
    }
    fn generate(&self, hit: &HitRecord, s: Sample2D, wi: Vec3) -> Option<Vec3> {
        match self {
            MaterialEnum::GGX(inner) => inner.generate(hit, s, wi),
            MaterialEnum::Lambertian(inner) => inner.generate(hit, s, wi),
            MaterialEnum::ParallelLight(inner) => inner.generate(hit, s, wi),
            MaterialEnum::DiffuseLight(inner) => inner.generate(hit, s, wi),
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
