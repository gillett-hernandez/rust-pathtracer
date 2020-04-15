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

pub trait Samplable {
    // method that should implement sampling a direction subtended by the solid angle of Self from point P
    fn sample(&self, point: Point3) -> Vec3;
    // method that should implement evaluating the pdf value of that sample having occurred, assuming random hemisphere sampling.
    fn pdf(&self, point: Point3, wi: Vec3) -> f32;
}
