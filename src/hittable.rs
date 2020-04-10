use crate::material::{BRDF, PDF};
use crate::materials::MaterialId;
use crate::math::*;

pub struct HitRecord {
    pub time: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material: Option<MaterialId>,
}

pub trait Hittable {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord>;
}

