use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;
#[derive(Debug)]
pub struct Lambertian {
    pub color: RGBColor,
}

impl Lambertian {
    pub fn new(color: RGBColor) -> Lambertian {
        Lambertian { color }
    }
}

impl PDF for Lambertian {
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        (wi * hit.normal).abs() / PI
    }
    fn generate(&self, hit: &HitRecord, s: &Box<dyn Sampler>, wi: Vec3) -> Vec3 {
        // random_cosine_direction(s)
        hit.normal + random_in_unit_sphere(s.draw_3d())
    }
}

impl BRDF for Lambertian {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        self.color * (wi * hit.normal).abs() / PI
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        RGBColor::ZERO
    }
}

impl Material for Lambertian {}
