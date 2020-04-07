use super::{BRDF, PDF};
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
    fn value(&self, wi: Vec3, wo: Vec3) -> f32 {
        1.0 / PI
    }
    fn generate(&self, s: Sample2D, wi: Vec3) -> Vec3 {
        random_cosine_direction(s)
    }
}

impl BRDF for Lambertian {
    fn f(&self, wi: Vec3, wo: Vec3) -> RGBColor {
        self.color / PI
    }
}
