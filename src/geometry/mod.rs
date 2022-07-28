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

macro_rules! generate_aggregate {
    ($($e:ident),+) => {
        #[derive(Clone, Debug)]
        pub enum Aggregate {
            $(
                $e($e),
            )+
        }
        $(
            impl From<$e> for Aggregate {
                fn from(data: $e) -> Self {
                    Aggregate::$e(data)
                }
            }
        )+


        impl HasBoundingBox for Aggregate {
            fn aabb(&self) -> AABB {
                match self {
                    $(
                        Self::$e(inner) => inner.aabb(),
                    )+
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
            $(
                Self::$e(inner) => inner.hit(r, t0, t1),
            )+
        }
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        debug_assert!(from.0.is_finite().all());
        let pair = match self {
            $(
                Self::$e(inner) => inner.sample(s, from),
            )+
        };
        debug_assert!((pair.0).0.is_finite().all(), "{:?} {:?}", self, pair.0);
        debug_assert!((pair.1).0.is_finite());
        pair
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let triplet = match self {
            $(
                Self::$e(inner) => inner.sample_surface(s),
            )+
        };
        debug_assert!((triplet.0).0.is_finite().all());
        debug_assert!((triplet.1).0.is_finite().all());
        debug_assert!((triplet.2).0.is_finite());
        triplet
    }
    fn psa_pdf(&self, cos_o: f32, from: Point3, to: Point3) -> PDF {
        debug_assert!(cos_o.is_finite());
        debug_assert!(from.0.is_finite().all());
        debug_assert!(to.0.is_finite().all());
        let pdf = match self {
            $(
                Self::$e(inner) => inner.psa_pdf(cos_o, from, to),
            )+
        };
        debug_assert!(
            pdf.0.is_finite(),
            "{:?}, {:?}, {:?}, {:?}, {}",
            self,
            cos_o,
            from,
            to,
            pdf.0
        );
        pdf
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        let res = match self {
            $(
                Self::$e(inner) => inner.surface_area(transform),
            )+
        };
        debug_assert!(res.is_finite());
        res
    }
}

    };
}

type Triangle = MeshTriangleRef;

generate_aggregate! {AARect, Sphere, Disk, Mesh, Triangle}
