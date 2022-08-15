use crate::prelude::*;
use std::fmt::Debug;

pub use crate::aabb::{HasBoundingBox, AABB};

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub time: f32,
    pub point: Point3,
    pub uv: (f32, f32),
    pub lambda: f32,
    pub normal: Vec3,
    pub material: MaterialId,
    pub instance_id: usize,
    pub transport_mode: TransportMode,
}

impl HitRecord {
    pub fn new(
        time: f32,
        point: Point3,
        uv: (f32, f32),
        lambda: f32,
        normal: Vec3,
        material: MaterialId,
        instance_id: usize,
        transport_mode: Option<TransportMode>,
    ) -> Self {
        HitRecord {
            time,
            point,
            uv,
            lambda,
            normal: normal.normalized(),
            material,
            instance_id,
            transport_mode: transport_mode.unwrap_or(TransportMode::Importance),
        }
    }
}
// impl Debug for HitRecord {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
//         write!(
//             f,
//             "time: {}, point: {:?}, normal: {:?}, material: {:?}, instance_id: {}",
//             self.time, self.point, self.normal, self.material, self.instance_id
//         )
//     }
// }

use std::marker::{Send, Sync};

pub trait Hittable: Send + Sync + HasBoundingBox {
    // unrelated to light sampling
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord>;

    // methods related to when the Hittable is an Emissive
    // method that should implement sampling a direction subtended by the solid angle of Self from point P
    // returns the solid angle PDF.
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF<f32, SolidAngle>);
    // method that should implement randomly sampling a point and normal on the surface of the object in object space
    // returns a point on the surface, the normal at that point, and the probability of that Point being chosen
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF<f32, Area>);

    // method that should implement the projected solid angle pdf of sampling this primitive from Vertex {from, normal}
    // to is on the surface of the hittable/light
    fn psa_pdf(&self, cos_o: f32, from: Point3, to: Point3) -> PDF<f32, ProjectedSolidAngle>;
    fn surface_area(&self, transform: &Transform3) -> f32;
}

// a supertrait of Hittable that allows indexing into it
// pub trait Indexable: Hittable {
//     fn get_primitive(&self, index: usize) -> &Box<dyn Hittable>;
// }
