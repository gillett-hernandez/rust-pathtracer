use crate::math::*;
use packed_simd::f32x4;

use std::ops::Mul;

pub trait HasBoundingBox {
    fn aabb(&self) -> AABB;
}

#[derive(Copy, Clone, Debug)]
pub struct AABB {
    pub min: Point3,
    pub max: Point3,
}

impl AABB {
    pub fn new(min: Point3, max: Point3) -> Self {
        AABB {
            min: Point3::from_raw(min.0.min(max.0)),
            max: Point3::from_raw(min.0.max(max.0)),
        }
    }
    // empty AABB contains nothing.
    pub fn empty() -> Self {
        AABB::new_raw(Point3::INFINITY, Point3::NEG_INFINITY)
    }
    pub const fn new_raw(min: Point3, max: Point3) -> Self {
        AABB { min, max }
    }

    pub fn contains(&self, point: Point3) -> bool {
        let min: f32x4 = self.min.0;
        let max: f32x4 = self.max.0;
        // point is only contained if its elements are all greater than or equal to the min and less than or equal to the max
        point.0.ge(min).all() && point.0.le(max).all()
    }

    pub fn hit(&self, r: &Ray, _t0: f32, _t1: f32) -> Option<(f32, f32)> {
        let denom = r.direction.0;
        const ZERO: f32x4 = f32x4::splat(0.0);

        let min: f32x4 = (denom.eq(ZERO)).select(ZERO, (self.min - r.origin).0 / denom);

        let max: f32x4 =
            (denom.eq(ZERO)).select(f32x4::splat(INFINITY), (self.max - r.origin).0 / denom);
        let tmin = min.min(max);
        let tmax = min.max(max);
        // println!("{:?} {:?}", tmin, tmax);
        if tmin.max_element() > tmax.min_element() {
            return None;
        }

        let scaled_t0 = (denom.eq(ZERO)).select(ZERO, f32x4::splat(_t0) / denom.abs());
        let scaled_t1 = f32x4::splat(_t1) / denom.abs();
        // println!("{:?} {:?}", scaled_t0, scaled_t1);
        if tmin.gt(scaled_t1).any() || tmax.lt(scaled_t0).any() {
            // if any of the "earliest" hit times (tmin) exceed the max allowable hit time for that dimension (scaled_t1), a hit cannot have occurred
            // if any of the "latest" hit times (tmax) are smaller than the minimum allowable hit time for that dimension (scaled_t0), a hit cannot have occurred.
            return None;
        }

        Some((
            scaled_t0.replace(3, f32::INFINITY).min_element(),
            scaled_t1.replace(3, f32::NEG_INFINITY).max_element(),
        ))

        // old code without simd.
        // let tmin: f32x4 = f32x4::splat(t0);
        // let tmax: f32x4 = f32x4::splat(t1);
        // assert that the absolute value of all the components of direction are greater than 0
        // assert!(
        //     r.direction.0.abs().gt(f32x4::splat(0.0)).all(),
        //     "{:?}",
        //     r.direction
        // );

        // return whether all of tmax's elements were greater than tmins
        // this can be made safe by replacing NaNs with positive or negative 1 depending on the side
        // tmax.gt(tmin).all()

        // t0 = ((self.min.x - r.origin.x()) / r.direction.x())
        //     .min((self.max.x() - r.origin.x()) / r.direction.x());
        // t1 = ((self.min.x() - r.origin.x()) / r.direction.x())
        //     .max((self.max.x() - r.origin.x()) / r.direction.x());
        // if tmax <= tmin {
        //     return false;
        // }
        // tmin = tmin.max(t0);
        // tmax = tmax.min(t1);

        // t0 = ((self.min.y() - r.origin.y()) / r.direction.y())
        //     .min((self.max.y() - r.origin.y()) / r.direction.y());
        // t1 = ((self.min.y() - r.origin.y()) / r.direction.y())
        //     .max((self.max.y() - r.origin.y()) / r.direction.y());
        // if tmax <= tmin {
        //     return false;
        // }
        // tmin = tmin.max(t0);
        // tmax = tmax.min(t1);

        // t0 = ((self.min.z() - r.origin.z()) / r.direction.z())
        //     .min((self.max.z() - r.origin.z()) / r.direction.z());
        // t1 = ((self.min.z() - r.origin.z()) / r.direction.z())
        //     .max((self.max.z() - r.origin.z()) / r.direction.z());
        // if tmax <= tmin {
        //     return false;
        // }
        // tmin = tmin.max(t0);
        // tmax = tmax.min(t1);
    }
    pub fn expand(mut self, other: &AABB) -> AABB {
        self.min = Point3::from_raw(self.min.0.min(other.min.0));
        self.max = Point3::from_raw(self.max.0.max(other.max.0));
        self
    }

    pub fn expand_mut(&mut self, other: &AABB) {
        self.min = Point3::from_raw(self.min.0.min(other.min.0));
        self.max = Point3::from_raw(self.max.0.max(other.max.0));
    }

    pub fn grow(mut self, other: &Point3) -> AABB {
        self.min = Point3::from_raw(self.min.0.min(other.0));
        self.max = Point3::from_raw(self.max.0.max(other.0));
        self
    }

    pub fn grow_mut(&mut self, other: &Point3) {
        self.min = Point3::from_raw(self.min.0.min(other.0));
        self.max = Point3::from_raw(self.max.0.max(other.0));
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn is_empty(&self) -> bool {
        self.min.0.gt(self.max.0).any()
    }

    pub fn center(&self) -> Point3 {
        self.min + (self.size() / 2.0)
    }

    pub fn surface_area(&self) -> f32 {
        let [sx, sy, sz, _]: [f32; 4] = self.size().0.into();
        2.0 * (sx * sy + sx * sz + sy * sz)
    }

    pub fn volume(&self) -> f32 {
        let [sx, sy, sz, _]: [f32; 4] = self.size().0.into();
        sx * sy * sz
    }

    // pub fn relative_eq(&self, other: &AABB, epsilon: f32) -> bool {
    //     relative_eq!(self.min, other.min, epsilon = epsilon)
    //         && relative_eq!(self.max, other.max, epsilon = epsilon)
    // }
}

impl Default for AABB {
    fn default() -> AABB {
        AABB::empty()
    }
}

impl Mul<AABB> for Matrix4x4 {
    type Output = AABB;
    fn mul(self, rhs: AABB) -> Self::Output {
        AABB::new(self * rhs.min, self * rhs.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_aabb() {
        let aabb1 = AABB::new(Point3::ORIGIN, Point3::new(1.0, 1.0, 1.0));
        println!("{:?}", aabb1);
        let test_ray = Ray::new(
            Point3::new(3.0, 3.0, 3.0),
            -Vec3::new(1.0, 1.0, 1.0).normalized(),
        );
        println!("{:?}", aabb1.hit(&test_ray, 0.2, 3.0));
        let test_ray = Ray::new(
            Point3::new(3.0, 3.0, 3.0),
            Vec3::new(1.0, 1.0, 1.0).normalized(),
        );
        println!("{:?}", aabb1.hit(&test_ray, 0.2, 3.0));
    }
}
