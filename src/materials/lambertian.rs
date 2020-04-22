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
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            cosine / PI
        } else {
            // -cosine / PI
            0.0
        }
        // cosine.abs() / PI
    }
    fn generate(&self, hit: &HitRecord, s: &mut Box<dyn Sampler>, wi: Vec3) -> Option<Vec3> {
        Some(random_cosine_direction(s.draw_2d()))
    }
}

impl BRDF for Lambertian {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        // 1.66018 * self.color / PI
        self.color / PI
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Option<Vec3>) -> RGBColor {
        RGBColor::ZERO
    }
}

impl Material for Lambertian {}
