// extern crate packed_simd;
mod bounds;
mod color;
mod misc;
mod point;
mod random;
mod sample;
pub mod spectral;
mod tangent_frame;
mod transform;
mod vec;

pub use bounds::Bounds1D;
pub use color::*;
pub use misc::*;
pub use point::Point3;
pub use random::*;
pub use sample::*;
pub use spectral::{
    InterpolationMode, SingleEnergy, SingleWavelength, SpectralPowerDistributionFunction, SPD,
};
pub use std::f32::consts::PI;
pub use std::f32::INFINITY;
pub use tangent_frame::TangentFrame;
pub use transform::Transform3;
pub use vec::{Axis, Vec3};

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone, PartialEq, Debug)]
pub enum Sidedness {
    Forward,
    Reverse,
    Dual,
}
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PDF(pub f32);
impl PDF {
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}

impl From<f32> for PDF {
    fn from(val: f32) -> Self {
        PDF(val)
    }
}

impl From<PDF> for f32 {
    fn from(val: PDF) -> Self {
        val.0
    }
}

impl Add for PDF {
    type Output = PDF;
    fn add(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 + rhs.0)
    }
}
impl AddAssign for PDF {
    fn add_assign(&mut self, rhs: PDF) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for PDF {
    type Output = PDF;
    fn mul(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 * rhs)
    }
}
impl Mul<PDF> for f32 {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self * rhs.0)
    }
}

impl Mul for PDF {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 * rhs.0)
    }
}

impl MulAssign for PDF {
    fn mul_assign(&mut self, other: PDF) {
        self.0 = self.0 * other.0
    }
}
impl Div<f32> for PDF {
    type Output = PDF;
    fn div(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 / rhs)
    }
}

#[derive(Copy, Clone, Debug)]
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
            tmax,
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
