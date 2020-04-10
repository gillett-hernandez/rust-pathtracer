use crate::hittable::HitRecord;
use crate::math::*;

pub trait PDF {
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32;
    fn generate(&self, hit: &HitRecord, s: &Box<dyn Sampler>, wi: Vec3) -> Vec3;
}
