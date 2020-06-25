use crate::materials::MaterialId;
use crate::math::*;
use std::fmt::Debug;

pub use crate::aabb::{HasBoundingBox, AABB};

pub struct HitRecord {
    pub time: f32,
    pub point: Point3,
    pub uv: (f32, f32),
    pub lambda: f32,
    pub normal: Vec3,
    pub material: Option<MaterialId>,
    pub instance_id: usize,
}

impl HitRecord {
    pub fn new(
        time: f32,
        point: Point3,
        uv: (f32, f32),
        lambda: f32,
        normal: Vec3,
        material: Option<MaterialId>,
        instance_id: usize,
    ) -> Self {
        HitRecord {
            time,
            point,
            uv,
            lambda,
            normal: normal.normalized(),
            material,
            instance_id,
        }
    }
}
impl Debug for HitRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "time: {}, point: {:?}, normal: {:?}, material: {:?}, instance_id: {}",
            self.time, self.point, self.normal, self.material, self.instance_id
        )
    }
}

use std::marker::{Send, Sync};

pub trait Hittable: Send + Sync + HasBoundingBox {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord>;
    // method that should implement sampling a direction subtended by the solid angle of Self from point P
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, f32);
    // method that should implement randomly sampling a point and normal on the surface of the object in object space
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3);
    // method that should implement evaluating the pdf value of that sample having occurred, assuming random hemisphere sampling.
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> f32;
    fn get_instance_id(&self) -> usize;
    fn get_material_id(&self) -> Option<MaterialId>;
}

// a supertrait of Hittable that allows indexing into it
// pub trait Indexable: Hittable {
//     fn get_primitive(&self, index: usize) -> &Box<dyn Hittable>;
// }
