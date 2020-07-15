mod disk;
mod instance;
mod mesh;
mod rect;
mod sphere;

pub use disk::Disk;
pub use instance::Instance;
pub use mesh::{Mesh, MeshTriangleRef};
pub use rect::AARect;
pub use sphere::Sphere;

use crate::hittable::{HasBoundingBox, HitRecord, Hittable, AABB};
use crate::math::*;

#[derive(Clone, Debug)]
pub enum Aggregate {
    AARect(AARect),
    Sphere(Sphere),
    Disk(Disk),
    Mesh(Mesh),
    Triangle(MeshTriangleRef),
}

impl From<Sphere> for Aggregate {
    fn from(data: Sphere) -> Self {
        Aggregate::Sphere(data)
    }
}

impl From<Disk> for Aggregate {
    fn from(data: Disk) -> Self {
        Aggregate::Disk(data)
    }
}

impl From<AARect> for Aggregate {
    fn from(data: AARect) -> Self {
        Aggregate::AARect(data)
    }
}

impl From<MeshTriangleRef> for Aggregate {
    fn from(data: MeshTriangleRef) -> Self {
        Aggregate::Triangle(data)
    }
}

impl HasBoundingBox for Aggregate {
    fn aabb(&self) -> AABB {
        match self {
            Aggregate::Sphere(inner) => inner.aabb(),
            Aggregate::AARect(inner) => inner.aabb(),
            Aggregate::Disk(inner) => inner.aabb(),
            Aggregate::Mesh(inner) => inner.aabb(),
            Aggregate::Triangle(inner) => inner.aabb(),
        }
    }
}

impl Hittable for Aggregate {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        debug_assert!(r.origin.0.is_finite().all());
        debug_assert!(r.direction.0.is_finite().all());
        debug_assert!(t0.is_finite());
        debug_assert!(!t1.is_nan(), "{:?} {:?} {:?}", r, t0, t1);

        match self {
            Aggregate::Sphere(inner) => inner.hit(r, t0, t1),
            Aggregate::Disk(inner) => inner.hit(r, t0, t1),
            Aggregate::AARect(inner) => inner.hit(r, t0, t1),
            Aggregate::Mesh(inner) => inner.hit(r, t0, t1),
            Aggregate::Triangle(inner) => inner.hit(r, t0, t1),
        }
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        debug_assert!(from.0.is_finite().all());
        let pair = match self {
            Aggregate::Sphere(inner) => inner.sample(s, from),
            Aggregate::Disk(inner) => inner.sample(s, from),
            Aggregate::AARect(inner) => inner.sample(s, from),
            Aggregate::Mesh(inner) => inner.sample(s, from),
            Aggregate::Triangle(inner) => inner.sample(s, from),
        };
        debug_assert!((pair.0).0.is_finite().all(), "{:?} {:?}", self, pair.0);
        debug_assert!((pair.1).0.is_finite());
        pair
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let triplet = match self {
            Aggregate::Sphere(inner) => inner.sample_surface(s),
            Aggregate::Disk(inner) => inner.sample_surface(s),
            Aggregate::AARect(inner) => inner.sample_surface(s),
            Aggregate::Mesh(inner) => inner.sample_surface(s),
            Aggregate::Triangle(inner) => inner.sample_surface(s),
        };
        debug_assert!((triplet.0).0.is_finite().all());
        debug_assert!((triplet.1).0.is_finite().all());
        debug_assert!((triplet.2).0.is_finite());
        triplet
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        debug_assert!(normal.0.is_finite().all());
        debug_assert!(from.0.is_finite().all());
        debug_assert!(to.0.is_finite().all());
        let pdf = match self {
            Aggregate::Sphere(inner) => inner.pdf(normal, from, to),
            Aggregate::Disk(inner) => inner.pdf(normal, from, to),
            Aggregate::AARect(inner) => inner.pdf(normal, from, to),
            Aggregate::Mesh(inner) => inner.pdf(normal, from, to),
            Aggregate::Triangle(inner) => inner.pdf(normal, from, to),
        };
        debug_assert!(
            pdf.0.is_finite(),
            "{:?}, {:?}, {:?}, {:?}",
            self,
            normal,
            from,
            to
        );
        pdf
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        let res = match self {
            Aggregate::Sphere(inner) => inner.surface_area(transform),
            Aggregate::Disk(inner) => inner.surface_area(transform),
            Aggregate::AARect(inner) => inner.surface_area(transform),
            Aggregate::Mesh(inner) => inner.surface_area(transform),
            Aggregate::Triangle(inner) => inner.surface_area(transform),
        };
        debug_assert!(res.is_finite());
        res
    }
}
