use crate::math::Vec3;
// use packed_simd::{f32x4, f32x8};
use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Debug)]
pub struct Point3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::new(x, y, z, 1.0))
    }
    pub const fn from_raw(v: f32x4) -> Point3 {
        Point3(v)
    }
    pub const ZERO: Point3 = Point3::from_raw(f32x4::splat(0.0));
    pub fn is_normal(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Point3 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn normalize(mut self) -> Self {
        unsafe {
            self.0 = self.0 / self.0.extract_unchecked(3);
        }
        self
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        Point3::from_raw(self.0 + other.0)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Point3::from_raw(self.0 - other.0)
    }
}

// // don't implement adding or subtracting floats from Point3, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3::from_raw((self.0 - other.0) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

impl From<[f32; 3]> for Point3 {
    fn from(other: [f32; 3]) -> Point3 {
        Point3::new(other[0], other[1], other[2])
    }
}
