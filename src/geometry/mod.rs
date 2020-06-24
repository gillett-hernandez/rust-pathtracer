mod cube;
mod rect;
mod sphere;

pub use rect::AARect;
pub use sphere::Sphere;

use crate::hittable::{HasBoundingBox, HitRecord, Hittable, AABB};
use crate::math::*;

pub enum Aggregate<'a> {
    AARect(AARect),
    Sphere(Sphere),
    Instance {
        aggregate: &'a Aggregate<'a>,
        transform: Transform3,
    },
}

impl Into<Aggregate<'static>> for Sphere {
    fn into(self) -> Aggregate<'static> {
        Aggregate::Sphere(self)
    }
}

impl Into<Aggregate<'static>> for AARect {
    fn into(self) -> Aggregate<'static> {
        Aggregate::AARect(self)
    }
}

impl<'a> Aggregate<'a> {
    fn wrap(&'a self, transform: Transform3) -> Aggregate<'a> {
        Aggregate::Instance {
            aggregate: &self,
            transform,
        }
    }
}

impl<'a> HasBoundingBox for Aggregate<'a> {
    fn bounding_box(&self) -> AABB {
        match self {
            Aggregate::Sphere(sphere) => sphere.bounding_box(),
            Aggregate::AARect(rect) => rect.bounding_box(),
            Aggregate::Instance {
                aggregate,
                transform,
            } => *transform * aggregate.bounding_box(),
        }
    }
}

impl<'a> Hittable for Aggregate<'a> {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        match self {
            Aggregate::Sphere(sphere) => sphere.hit(r, t0, t1),
            Aggregate::AARect(rect) => rect.hit(r, t0, t1),
            Aggregate::Instance {
                aggregate,
                transform,
            } => {
                if let Some(hit) = aggregate.hit(*transform / r, t0, t1) {
                    Some(HitRecord {
                        normal: *transform * hit.normal,
                        point: *transform * hit.point,
                        ..hit
                    })
                } else {
                    None
                }
            }
        }
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, f32) {
        match self {
            Aggregate::Sphere(sphere) => sphere.sample(s, from),
            Aggregate::AARect(rect) => rect.sample(s, from),
            Aggregate::Instance {
                aggregate,
                transform,
            } => {
                let (vec, pdf) = aggregate.sample(s, *transform / from);
                (*transform * vec, pdf)
            }
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> f32 {
        match self {
            Aggregate::Sphere(sphere) => sphere.pdf(normal, from, to),
            Aggregate::AARect(rect) => rect.pdf(normal, from, to),
            Aggregate::Instance {
                aggregate,
                transform,
            } => aggregate.pdf(*transform / normal, *transform / from, *transform / to),
        }
    }
    fn get_instance_id(&self) -> usize {
        match self {
            Aggregate::Sphere(sphere) => sphere.get_instance_id(),
            Aggregate::AARect(rect) => rect.get_instance_id(),
            Aggregate::Instance {
                aggregate,
                transform: _,
            } => aggregate.get_instance_id(),
        }
    }
}
