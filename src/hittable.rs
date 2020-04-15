use crate::material::{BRDF, PDF};
use crate::materials::MaterialId;
use crate::math::*;

pub struct HitRecord {
    pub time: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material: Option<MaterialId>,
}

impl HitRecord {
    pub fn new(time: f32, point: Point3, normal: Vec3, material: Option<MaterialId>) -> Self {
        HitRecord {
            time,
            point,
            normal: normal.normalized(),
            material,
        }
    }
}

pub trait Hittable {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord>;
}
