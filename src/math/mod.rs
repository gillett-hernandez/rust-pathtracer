// extern crate packed_simd;
mod misc;
mod point;
mod vec;
pub use misc::*;
pub use point::Point3;
pub use std::f32::consts::PI;
pub use std::f32::INFINITY;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
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
    fn to_rgb(&self) -> [f32; 3];
}
#[derive(Copy, Clone)]
pub struct RGBColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RGBColor {
    pub const fn new(r: f32, g: f32, b: f32) -> RGBColor {
        RGBColor { r, g, b }
    }
    pub const ZERO: RGBColor = RGBColor::new(0.0, 0.0, 0.0);
}

impl Color for RGBColor {
    fn to_rgb(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }
}

impl AddAssign for RGBColor {
    fn add_assign(&mut self, other: RGBColor) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}
impl DivAssign<f32> for RGBColor {
    fn div_assign(&mut self, other: f32) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
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
