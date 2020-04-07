use crate::math::*;

pub trait BRDF {
    fn f(&self, wi: Vec3, wo: Vec3) -> RGBColor;
}
