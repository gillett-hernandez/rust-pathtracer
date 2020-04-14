use crate::math::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign};
#[derive(Copy, Clone, Debug)]
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
    fn to_rgb(&self) -> [u8; 3] {
        [self.r as u8, self.g as u8, self.b as u8]
    }
}

impl Add for RGBColor {
    type Output = RGBColor;
    fn add(self, other: RGBColor) -> RGBColor {
        RGBColor::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }
}

impl AddAssign for RGBColor {
    fn add_assign(&mut self, other: RGBColor) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}
impl Div<f32> for RGBColor {
    type Output = RGBColor;
    fn div(mut self, other: f32) -> Self {
        self.r /= other;
        self.g /= other;
        self.b /= other;
        self
    }
}

impl DivAssign<f32> for RGBColor {
    fn div_assign(&mut self, other: f32) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
    }
}

impl Mul for RGBColor {
    type Output = RGBColor;
    fn mul(self, other: RGBColor) -> RGBColor {
        RGBColor::new(self.r * other.r, self.g * other.g, self.b * other.b)
    }
}

impl MulAssign for RGBColor {
    fn mul_assign(&mut self, other: RGBColor) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
    }
}

impl Mul<f32> for RGBColor {
    type Output = RGBColor;
    fn mul(self, other: f32) -> RGBColor {
        RGBColor::new(self.r * other, self.g * other, self.b * other)
    }
}

impl Mul<RGBColor> for f32 {
    type Output = RGBColor;
    fn mul(self, other: RGBColor) -> RGBColor {
        RGBColor::new(other.r * self, other.g * self, other.b * self)
    }
}
