use crate::hittable::HitRecord;
use crate::math::*;

pub trait BRDF {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor;
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> RGBColor;
}
