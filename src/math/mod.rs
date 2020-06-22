// extern crate packed_simd;
mod bounds;
mod color;
mod misc;
mod point;
mod random;
mod sample;
pub mod spectral;
mod tangent_frame;
mod vec;

pub use bounds::Bounds1D;
pub use color::*;
pub use misc::*;
pub use point::Point3;
pub use random::*;
pub use sample::*;
pub use spectral::{SingleEnergy, SingleWavelength, SpectralPowerDistributionFunction, SPD};
pub use std::f32::consts::PI;
pub use std::f32::INFINITY;
pub use tangent_frame::TangentFrame;
pub use vec::{Axis, Vec3};

use std::fmt::Debug;
use std::ops::Mul;

#[derive(Copy, Clone, PartialEq)]
pub enum Sidedness {
    Forward,
    Reverse,
    Dual,
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
