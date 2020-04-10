// extern crate packed_simd;
mod color;
mod misc;
mod point;
mod sample;
mod vec;
pub use color::RGBColor;
pub use misc::*;
pub use point::Point3;
pub use sample::*;
pub use std::f32::consts::PI;
pub use std::f32::INFINITY;
use std::ops::Mul;
pub use vec::Vec3;

impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        Vec3::new(p.x, p.y, p.z)
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        Point3::new(v.x, v.y, v.z)
    }
}

pub trait Color {
    fn to_rgb(&self) -> [u8; 3];
}

impl Mul<RGBColor> for Vec3 {
    type Output = RGBColor;
    fn mul(mut self, other: RGBColor) -> RGBColor {
        self.x *= other.r;
        self.y *= other.g;
        self.z *= other.b;
        RGBColor::new(self.x, self.y, self.z)
    }
}

impl From<RGBColor> for Vec3 {
    fn from(c: RGBColor) -> Vec3 {
        Vec3::new(c.r, c.g, c.b)
    }
}

#[derive(Copy, Clone)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f32,
    pub tmax: f32,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    pub const fn new_with_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    pub const fn new_with_time_and_tmax(
        origin: Point3,
        direction: Vec3,
        time: f32,
        tmax: f32,
    ) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: tmax,
        }
    }
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
    // pub fn at_time(mut self, time: f32) -> Self {
    //     // self.origin =
    // }
    pub fn point_at_parameter(self, time: f32) -> Point3 {
        self.origin + self.direction * time
    }
}
