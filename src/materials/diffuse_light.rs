use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;
pub struct DiffuseLight {
    pub color: SDF,
}

impl DiffuseLight {
    pub fn new(color: SDF) -> DiffuseLight {
        DiffuseLight { color }
    }
}

impl PDF for DiffuseLight {
    // PDF has a different meaning for light sources.
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        0.0
    }
    fn generate(&self, hit: &HitRecord, s: &mut Box<dyn Sampler>, wi: Vec3) -> Option<Vec3> {
        None
    }
}

impl BRDF for DiffuseLight {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> SingleEnergy {
        SingleEnergy::ZERO
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Option<Vec3>) -> SingleEnergy {
        SingleEnergy::new(self.color.evaluate_power(hit.lambda))
    }
}

impl Material for DiffuseLight {}
