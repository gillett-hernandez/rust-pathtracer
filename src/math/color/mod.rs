use crate::math::misc::gaussian;
use crate::math::*;

use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign};

use nalgebra::{Matrix3, Vector3};

mod rgb;
mod xyz;
pub use rgb::RGBColor;
pub use xyz::XYZColor;

impl From<XYZColor> for RGBColor {
    fn from(xyz: XYZColor) -> Self {
        let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
            0.41847, -0.15866, -0.082835, -0.091169, 0.25243, 0.015708, 0.00092090, -0.0025498,
            0.17860,
        );
        let [a, b, c, _]: [f32; 4] = xyz.0.into();
        let intermediate = xyz_to_rgb * Vector3::new(a, b, c);
        RGBColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}

impl From<RGBColor> for XYZColor {
    fn from(rgb: RGBColor) -> Self {
        let rgb_to_xyz: Matrix3<f32> = Matrix3::new(
            0.490, 0.310, 0.200, 0.17697, 0.8124, 0.01063, 0.0, 0.01, 0.99,
        );
        let [a, b, c, _]: [f32; 4] = rgb.0.into();
        let intermediate = rgb_to_xyz * Vector3::new(a, b, c) / 0.17697;
        XYZColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}
