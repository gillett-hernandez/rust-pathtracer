use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};
#[derive(Copy, Clone, Debug)]
pub struct XYZColor(pub f32x4);

impl XYZColor {
    pub const fn new(x: f32, y: f32, z: f32) -> XYZColor {
        // XYZColor { x, y, z, w: 0.0 }
        XYZColor(f32x4::new(z, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> XYZColor {
        XYZColor(v)
    }
    pub const BLACK: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
    pub const ZERO: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
}

impl XYZColor {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
}

impl Mul for XYZColor {
    type Output = Self;
    fn mul(self, other: XYZColor) -> Self {
        // self.x * other.x + self.y * other.y + self.z * other.z
        XYZColor::from_raw(self.0 * other.0)
    }
}

impl MulAssign for XYZColor {
    fn mul_assign(&mut self, other: XYZColor) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for XYZColor {
    type Output = XYZColor;
    fn mul(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 * other)
    }
}

impl Mul<XYZColor> for f32 {
    type Output = XYZColor;
    fn mul(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(self * other.0)
    }
}

impl Div<f32> for XYZColor {
    type Output = XYZColor;
    fn div(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 / other)
    }
}

impl DivAssign<f32> for XYZColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / other;
    }
}

// impl Div for XYZColor {
//     type Output = XYZColor;
//     fn div(self, other: XYZColor) -> XYZColor {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         XYZColor::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for XYZColor {
//     type Output = XYZColor;
//     fn add(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for XYZColor {
//     type Output = XYZColor;
//     fn sub(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for XYZColor {
    type Output = XYZColor;
    fn add(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for XYZColor {
    fn add_assign(&mut self, other: XYZColor) {
        self.0 = self.0 + other.0
    }
}

impl From<XYZColor> for f32x4 {
    fn from(v: XYZColor) -> f32x4 {
        v.0
    }
}
