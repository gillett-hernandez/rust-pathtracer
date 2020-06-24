use crate::aabb::AABB;
use crate::math::*;

use nalgebra;
use packed_simd::{f32x16, f32x4};
use std::ops::{Div, Mul, MulAssign};

#[derive(Debug, Copy, Clone)]
pub struct Matrix4x4(f32x16);

impl Mul<Vec3> for Matrix4x4 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        unimplemented!()
    }
}

impl Mul<Point3> for Matrix4x4 {
    type Output = Point3;
    fn mul(self, rhs: Point3) -> Self::Output {
        unimplemented!()
    }
}

impl Div<Vec3> for Matrix4x4 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        unimplemented!()
    }
}

impl Div<Point3> for Matrix4x4 {
    type Output = Point3;
    fn div(self, rhs: Point3) -> Self::Output {
        unimplemented!()
    }
}

impl Mul for Matrix4x4 {
    type Output = Matrix4x4;
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        unimplemented!()
    }
}

impl Div for Matrix4x4 {
    type Output = Matrix4x4;
    fn div(self, rhs: Matrix4x4) -> Self::Output {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Transform3 {
    pub forward: Matrix4x4,
    pub reverse: Matrix4x4,
}

impl Transform3 {
    pub fn new_from_matrix(forward: nalgebra::Matrix4<f32>) -> Self {
        Transform3 {
            forward: Matrix4x4::from(forward),
            reverse: Matrix4x4::from(forward.try_inverse().expect("matrix inverse failed")),
        }
    }

    pub fn translation(shift: Vec3) -> Self {
        unimplemented!()
    }

    pub fn scale(scale: Vec3) -> Self {
        unimplemented!()
    }

    pub fn axis_angle(axis: Vec3, radians: f32) -> Self {
        unimplemented!()
    }

    pub fn rotation(quaternion: f32x4) -> Self {
        unimplemented!()
    }

    pub fn new_from_raw(forward: Matrix4x4, reverse: Matrix4x4) -> Self {
        Transform3 { forward, reverse }
    }
}

impl From<nalgebra::Matrix4<f32>> for Matrix4x4 {
    fn from(matrix: nalgebra::Matrix4<f32>) -> Self {
        unimplemented!()
    }
}

impl From<Matrix4x4> for nalgebra::Matrix4<f32> {
    fn from(matrix: Matrix4x4) -> Self {
        unimplemented!()
    }
}

impl Mul<Vec3> for Transform3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        self.forward * rhs
    }
}

impl Mul<Point3> for Transform3 {
    type Output = Point3;
    fn mul(self, rhs: Point3) -> Self::Output {
        self.forward * rhs
    }
}

impl Mul<Ray> for Transform3 {
    type Output = Ray;
    fn mul(self, rhs: Ray) -> Self::Output {
        Ray {
            origin: self * rhs.origin,
            direction: self * rhs.direction,
            ..rhs
        }
    }
}

impl Mul<AABB> for Transform3 {
    type Output = AABB;
    fn mul(self, rhs: AABB) -> Self::Output {
        AABB::new(self * rhs.min, self * rhs.max)
    }
}

impl Mul<Transform3> for Transform3 {
    type Output = Transform3;
    fn mul(self, rhs: Transform3) -> Self::Output {
        Transform3::new_from_raw(rhs.forward * self.forward, rhs.reverse * self.reverse)
    }
}

impl Div<Vec3> for Transform3 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        self.reverse * rhs
    }
}

impl Div<Point3> for Transform3 {
    type Output = Point3;
    fn div(self, rhs: Point3) -> Self::Output {
        self.reverse * rhs
    }
}

impl Div<Ray> for Transform3 {
    type Output = Ray;
    fn div(self, rhs: Ray) -> Self::Output {
        Ray {
            origin: self / rhs.origin,
            direction: self / rhs.direction,
            ..rhs
        }
    }
}

impl Div<AABB> for Transform3 {
    type Output = AABB;
    fn div(self, rhs: AABB) -> Self::Output {
        AABB::new(self / rhs.min, self / rhs.max)
    }
}

impl Div<Transform3> for Transform3 {
    type Output = Transform3;
    fn div(self, rhs: Transform3) -> Self::Output {
        Transform3::new_from_raw(rhs.reverse / self.reverse, rhs.forward * self.forward)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
