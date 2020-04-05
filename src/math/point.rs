use crate::math::Vec3;
use std::ops::{Add, Mul, Neg, Sub};
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3 { x, y, z, w: 1.0 }
    }
    pub const ZERO: Point3 = Point3::new(0.0, 0.0, 0.0);
    pub const X: Point3 = Point3::new(1.0, 0.0, 0.0);
    pub const Y: Point3 = Point3::new(0.0, 1.0, 0.0);
    pub const Z: Point3 = Point3::new(0.0, 0.0, 1.0);
}

impl Add<f32> for Point3 {
    type Output = Point3;
    fn add(self, other: f32) -> Point3 {
        Point3::new(self.x + other, self.y + other, self.z + other)
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub<f32> for Point3 {
    type Output = Point3;
    fn sub(self, other: f32) -> Point3 {
        Point3::new(self.x - other, self.y - other, self.z - other)
    }
}

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
