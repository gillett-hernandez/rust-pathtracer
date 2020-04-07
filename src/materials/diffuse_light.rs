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
    fn value(&self, wi: Vec3, wo: Vec3) -> f32 {
        1.0 / PI
    }
    fn generate(&self, s: Sample2D, wi: Vec3) -> Vec3 {
        random_cosine_direction(s)
    }
}

impl BRDF for DiffuseLight {
    fn f(&self, wi: Vec3, wo: Vec3) -> RGBColor {
        RGBColor::ZERO
    }
    fn emission(&self, wi: Vec3, wo: Vec3) -> RGBColor {
        self.color
    }
}

impl Material for DiffuseLight {}
