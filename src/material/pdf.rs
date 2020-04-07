use crate::math::*;

pub trait PDF {
    fn value(&self, wi: Vec3, wo: Vec3) -> f32;
    fn generate(&self, s: Sample2D, wi: Vec3) -> Vec3;
}
