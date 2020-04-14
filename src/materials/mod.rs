// pub use crate::material::{BRDF, BTDF, BXDF, PDF};
use crate::hittable::HitRecord;
pub use crate::material::{Material, BRDF, PDF};
use crate::math::*;

pub mod diffuse_light;
pub mod lambertian;
pub use diffuse_light::DiffuseLight;
pub use lambertian::Lambertian;

// type required for an id into the Material Table
pub type MaterialId = u8;

// pub struct MaterialTable {
//     pub materials: Vec<Box<dyn Material>>,
// }
pub type MaterialTable = Vec<Box<dyn Material>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambertian() {
        let lambertian = Lambertian::new(RGBColor::new(0.9, 0.2, 0.9));
        // simulate incoming ray from directly above
        let incoming: Ray = Ray::new(Point3::new(0.0, 0.0, 10.0), -Vec3::Z);
        let hit = HitRecord {
            time: 0.0,
            point: Point3::ZERO,
            normal: Vec3::Z,
            material: Some(0),
        };
        let sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let v = lambertian.generate(&hit, &sampler, incoming.direction);
        println!("{:?}", lambertian.f(&hit, incoming.direction, v));
        assert!(lambertian.value(&hit, incoming.direction, v) > 0.0);
        println!("{:?}", v);
        assert!(v * Vec3::Z > 0.0);
    }
    #[test]
    fn test_lambertian_integral() {
        let lambertian = Lambertian::new(RGBColor::new(1.0, 1.0, 1.0));
        // simulate incoming ray from directly above
        let incoming: Ray = Ray::new(Point3::new(0.0, 0.0, 10.0), -Vec3::Z);
        let hit = HitRecord {
            time: 0.0,
            point: Point3::ZERO,
            normal: Vec3::Z,
            material: Some(0),
        };
        let sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let mut pdf_sum = 0.0;
        let mut color_sum = RGBColor::ZERO;
        let N = 1000000;
        for i in 0..N {
            let v = lambertian.generate(&hit, &sampler, -incoming.direction);
            let reflectance = lambertian.f(&hit, -incoming.direction, v);
            let pdf = lambertian.value(&hit, -incoming.direction, v);
            assert!(pdf > 0.0);
            assert!(v * Vec3::Z > 0.0);
            pdf_sum += pdf;
            color_sum += reflectance;
        }
        println!("{:?}", color_sum / N as f32);
        println!("{:?}", pdf_sum / N as f32);
        assert!((pdf_sum / N as f32 - 1.0).abs() < 10000.0 / N as f32);
    }
}
