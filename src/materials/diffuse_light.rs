use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;
#[derive(Debug)]
pub struct DiffuseLight {
    pub color: RGBColor,
}

impl DiffuseLight {
    pub fn new(color: RGBColor) -> DiffuseLight {
        DiffuseLight { color }
    }
}

impl PDF for DiffuseLight {
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        0.0
    }
    fn generate(&self, hit: &HitRecord, s: &Box<dyn Sampler>, wi: Vec3) -> Vec3 {
        Vec3::ZERO
    }
}

impl BRDF for DiffuseLight {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        RGBColor::ZERO
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor {
        self.color
    }
}

impl Material for DiffuseLight {}
