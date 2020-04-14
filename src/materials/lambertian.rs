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
        let cosine = wo.normalized() * hit.normal;
        if cosine > 0.0 {
            // in same hemisphere
            // (cosine).abs()
            cosine / PI // works with the "decently good" one
        } else {
            0.0
        }
    }
    fn generate(&self, hit: &HitRecord, s: &Box<dyn Sampler>, wi: Vec3) -> Vec3 {
        // random_cosine_direction(s)
        // (hit.normal + random_in_unit_sphere(s.draw_3d())).normalized()

        let frame = TangentFrame::from_normal(hit.normal);
        // let wi_local = frame.to_local(wi);
        frame
            .to_local(&random_cosine_direction(s.draw_2d()))
            .normalized()
    }
}

impl BRDF for Lambertian {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        // (wi * hit.normal).abs() * self.color // fireflies, no good
        // self.color // fireflies
        // (wo * hit.normal).abs() * self.color / PI // dark
        assert!(wi * hit.normal > 0.0, "aligned is what we want, and wi (which is the negation of the incoming rays direction) and hit.normal were not aligned. {:?}.{:?}", wi, hit.normal);
        self.color / PI // fireflies
                        // (wo * hit.normal).abs() * self.color // decently good
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        RGBColor::ZERO
    }
}

impl Material for Lambertian {}
