use crate::prelude::*;

use std::{ops::Mul, simd::Mask};

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
            min: Point3(min.0.min(max.0)),
            max: Point3(min.0.max(max.0)),
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
        point.0.simd_ge(min).all() && point.0.simd_le(max).all()
    }

    pub fn hit(&self, r: &Ray, _t0: f32, _t1: f32) -> Option<(f32, f32)> {
        let denom = r.direction.0;
        const ZERO: f32x4 = f32x4::ZERO;

        let min: f32x4 = (denom.simd_eq(ZERO)).select(ZERO, (self.min - r.origin).0 / denom);

        let max: f32x4 =
            (denom.simd_eq(ZERO)).select(f32x4::splat(INFINITY), (self.max - r.origin).0 / denom);
        let tmin = min.simd_min(max);
        let tmax = min.simd_max(max);
        if tmin.reduce_max() > tmax.reduce_min() {
            return None;
        }

        let mut scaled_t0 =
            (denom.simd_eq(ZERO)).select(ZERO, f32x4::splat(_t0) / SimdFloat::abs(denom));
        let mut scaled_t1 = f32x4::splat(_t1) / SimdFloat::abs(denom);
        // println!("{:?} {:?}", scaled_t0, scaled_t1);
        if tmin.simd_gt(scaled_t1).any() || tmax.simd_lt(scaled_t0).any() {
            // if any of the "earliest" hit times (tmin) exceed the max allowable hit time for that dimension (scaled_t1), a hit cannot have occurred
            // if any of the "latest" hit times (tmax) are smaller than the minimum allowable hit time for that dimension (scaled_t0), a hit cannot have occurred.
            return None;
        }

        scaled_t0[3] = f32::INFINITY;
        scaled_t1[3] = f32::NEG_INFINITY;

        Some((scaled_t0.reduce_min(), scaled_t1.reduce_max()))
    }
    pub fn expand(mut self, other: &AABB) -> AABB {
        self.min = Point3(self.min.0.min(other.min.0));
        self.max = Point3(self.max.0.max(other.max.0));
        self
    }

    pub fn expand_mut(&mut self, other: &AABB) {
        self.min = Point3(self.min.0.min(other.min.0));
        self.max = Point3(self.max.0.max(other.max.0));
    }

    pub fn grow(mut self, other: &Point3) -> AABB {
        self.min = Point3(self.min.0.min(other.0));
        self.max = Point3(self.max.0.max(other.0));
        self
    }

    pub fn grow_mut(&mut self, other: &Point3) {
        self.min = Point3(self.min.0.min(other.0));
        self.max = Point3(self.max.0.max(other.0));
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn is_empty(&self) -> bool {
        self.min.0.simd_gt(self.max.0).any()
    }

    pub fn center(&self) -> Point3 {
        self.min + (self.size() / 2.0)
    }

    pub fn surface_area(&self) -> f32 {
        let [sx, sy, sz, _]: [f32; 4] = self.size().as_array();
        2.0 * (sx * sy + sx * sz + sy * sz)
    }

    pub fn volume(&self) -> f32 {
        let [sx, sy, sz, _]: [f32; 4] = self.size().as_array();
        sx * sy * sz
    }
}

impl Default for AABB {
    fn default() -> AABB {
        AABB::empty()
    }
}

impl Mul<AABB> for Matrix4x4 {
    type Output = AABB;
    fn mul(self, rhs: AABB) -> Self::Output {
        // need to transform all 8 corner points and make a bounding box surrounding all of them.
        // all 8 corner points can be reconstructed by this procedure:

        let mut min = f32x4::splat(f32::INFINITY);
        let mut max = f32x4::splat(-f32::INFINITY);

        min[3] = 1.0;
        max[3] = 1.0;

        for index in 0..8 {
            let (xb, yb, zb) = (index & 1 == 0, (index >> 1) & 1 == 0, (index >> 2) & 1 == 0);
            let candidate = (self
                * Point3(Mask::from_array([xb, yb, zb, false]).select(rhs.min.0, rhs.max.0)))
            .0;
            min = min.min(candidate);
            max = max.max(candidate);
        }
        AABB::new_raw(Point3(min), Point3(max))
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
