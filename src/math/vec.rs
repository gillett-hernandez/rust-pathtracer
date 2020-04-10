// use packed_simd::{f32x4, f32x8};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z, w: 0.0 }
    }
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
}

impl Mul for Vec3 {
    type Output = f32;
    fn mul(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3::new(self.x * other, self.y * other, self.z * other)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        other * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3::new(self.x / other, self.y / other, self.z / other)
    }
}
// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3::new(s, s, s)
    }
}

impl Vec3 {
    pub fn cross(&self, other: Vec3) -> Self {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm_squared(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3::new(self.x / norm, self.y / norm, self.z / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec() {
        let v = Vec3::new(100.0, 0.2, 1.0);
        assert!(v.norm() > 100.0);
        assert!(v.norm_squared() > 10000.0);
        assert!(v.normalized().norm() - 1.0 < 0.000001);
    }
}
