use crate::math::*;

use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign};

// impl Color for RGBColor {
//     fn to_rgb(&self) -> [u8; 3] {

//         let [r, g, b, _]: [f32; 4] = color.0.into();
//         [self.r as u8, self.g as u8, self.b as u8]
//     }
// }

#[derive(Copy, Clone, Debug)]
pub struct RGBColor(pub f32x4);

impl RGBColor {
    pub const fn new(r: f32, g: f32, b: f32) -> RGBColor {
        // RGBColor { x, y, z, w: 0.0 }
        RGBColor(f32x4::new(r, g, b, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> RGBColor {
        RGBColor(v)
    }
    pub const ZERO: RGBColor = RGBColor::from_raw(f32x4::splat(0.0));
}

impl RGBColor {
    #[inline(always)]
    pub fn r(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn g(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn b(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
}

impl Mul for RGBColor {
    type Output = Self;
    fn mul(self, other: RGBColor) -> Self {
        // self.x * other.x + self.y * other.y + self.z * other.z
        RGBColor::from_raw(self.0 * other.0)
    }
}

impl MulAssign for RGBColor {
    fn mul_assign(&mut self, other: RGBColor) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for RGBColor {
    type Output = RGBColor;
    fn mul(self, other: f32) -> RGBColor {
        RGBColor::from_raw(self.0 * other)
    }
}

impl Mul<RGBColor> for f32 {
    type Output = RGBColor;
    fn mul(self, other: RGBColor) -> RGBColor {
        RGBColor::from_raw(self * other.0)
    }
}

impl Div<f32> for RGBColor {
    type Output = RGBColor;
    fn div(self, other: f32) -> RGBColor {
        RGBColor::from_raw(self.0 / other)
    }
}

impl DivAssign<f32> for RGBColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / other;
    }
}

// impl Div for RGBColor {
//     type Output = RGBColor;
//     fn div(self, other: RGBColor) -> RGBColor {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         RGBColor::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for RGBColor {
//     type Output = RGBColor;
//     fn add(self, other: f32) -> RGBColor {
//         RGBColor::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for RGBColor {
//     type Output = RGBColor;
//     fn sub(self, other: f32) -> RGBColor {
//         RGBColor::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for RGBColor {
    type Output = RGBColor;
    fn add(self, other: RGBColor) -> RGBColor {
        RGBColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for RGBColor {
    fn add_assign(&mut self, other: RGBColor) {
        self.0 = self.0 + other.0
    }
}

impl From<f32> for RGBColor {
    fn from(s: f32) -> RGBColor {
        RGBColor::from_raw(f32x4::splat(s) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

impl From<RGBColor> for f32x4 {
    fn from(v: RGBColor) -> f32x4 {
        v.0
    }
}
