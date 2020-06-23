use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;
pub struct Lambertian {
    pub color: SPD,
}

impl Lambertian {
    pub fn new(color: SPD) -> Lambertian {
        Lambertian { color }
    }
}

impl PDF for Lambertian {
    fn value(&self, _hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            cosine / PI
        } else {
            0.0
        }
    }
    fn generate(&self, _hit: &HitRecord, s: &mut Box<dyn Sampler>, _wi: Vec3) -> Option<Vec3> {
        Some(random_cosine_direction(s.draw_2d()))
    }
}

impl BRDF for Lambertian {
    fn f(&self, hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> SingleEnergy {
        SingleEnergy::new(self.color.evaluate(hit.lambda) / PI)
    }
    fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
        SingleEnergy::ZERO
    }
}

impl Material for Lambertian {}
