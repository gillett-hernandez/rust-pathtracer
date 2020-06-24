use crate::aabb::AABB;
use crate::math::*;

use nalgebra;
use packed_simd::{f32x16, f32x4};
use std::ops::{Div, Mul};

#[derive(Debug, Copy, Clone)]
pub struct Matrix4x4(f32x16);

// impl Matrix4x4 {
//     const I: Matrix4x4 = Matrix4x4(f32x16::new(
//         1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
//     ));
// }

impl Mul<Vec3> for Matrix4x4 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        let [v0, v1, v2, v3]: [f32; 4] = rhs.0.into();

        let column0: f32x4 = shuffle!(self.0, [0, 4, 8, 12]);
        let column1: f32x4 = shuffle!(self.0, [1, 5, 9, 13]);
        let column2: f32x4 = shuffle!(self.0, [2, 6, 10, 14]);
        let column3: f32x4 = shuffle!(self.0, [3, 7, 11, 15]);

        let result = column0 * v0 + column1 * v1 + column2 * v2 + column3 * v3;

        Vec3::from_raw(result)
    }
}

impl Mul<Point3> for Matrix4x4 {
    type Output = Point3;
    fn mul(self, rhs: Point3) -> Self::Output {
        // only apply scale and rotation
        let [v0, v1, v2, v3]: [f32; 4] = rhs.0.into();

        let column0: f32x4 = shuffle!(self.0, [0, 4, 8, 12]);
        let column1: f32x4 = shuffle!(self.0, [1, 5, 9, 13]);
        let column2: f32x4 = shuffle!(self.0, [2, 6, 10, 14]);
        let column3: f32x4 = shuffle!(self.0, [3, 7, 11, 15]);

        let result = column0 * v0 + column1 * v1 + column2 * v2 + column3 * v3;

        Point3::from_raw(result)
    }
}

impl Mul for Matrix4x4 {
    type Output = Matrix4x4;
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        // let a_column1: f32x4 = shuffle!(self.0, [0, 4, 8, 12]);
        // let a_column2: f32x4 = shuffle!(self.0, [1, 5, 9, 13]);
        // let a_column3: f32x4 = shuffle!(self.0, [2, 6, 10, 14]);
        // let a_column4: f32x4 = shuffle!(self.0, [3, 7, 11, 15]);

        let a_row1: f32x4 = shuffle!(self.0, [0, 1, 2, 3]);
        let a_row2: f32x4 = shuffle!(self.0, [4, 5, 6, 7]);
        let a_row3: f32x4 = shuffle!(self.0, [8, 9, 10, 11]);
        let a_row4: f32x4 = shuffle!(self.0, [12, 13, 14, 15]);

        let b_column1: f32x4 = shuffle!(rhs.0, [0, 4, 8, 12]);
        let b_column2: f32x4 = shuffle!(rhs.0, [1, 5, 9, 13]);
        let b_column3: f32x4 = shuffle!(rhs.0, [2, 6, 10, 14]);
        let b_column4: f32x4 = shuffle!(rhs.0, [3, 7, 11, 15]);

        // let b_row1: f32x4 = shuffle!(rhs.0, [0, 1, 2, 3]);
        // let b_row2: f32x4 = shuffle!(rhs.0, [4, 5, 6, 7]);
        // let b_row3: f32x4 = shuffle!(rhs.0, [8, 9, 10, 11]);
        // let b_row4: f32x4 = shuffle!(rhs.0, [12, 13, 14, 15]);

        let m11 = (a_row1 * b_column1).sum();
        let m12 = (a_row1 * b_column2).sum();
        let m13 = (a_row1 * b_column3).sum();
        let m14 = (a_row1 * b_column4).sum();

        let m21 = (a_row2 * b_column1).sum();
        let m22 = (a_row2 * b_column2).sum();
        let m23 = (a_row2 * b_column3).sum();
        let m24 = (a_row2 * b_column4).sum();

        let m31 = (a_row3 * b_column1).sum();
        let m32 = (a_row3 * b_column2).sum();
        let m33 = (a_row3 * b_column3).sum();
        let m34 = (a_row3 * b_column4).sum();

        let m41 = (a_row4 * b_column1).sum();
        let m42 = (a_row4 * b_column2).sum();
        let m43 = (a_row4 * b_column3).sum();
        let m44 = (a_row4 * b_column4).sum();

        Matrix4x4 {
            0: f32x16::new(
                m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44,
            ),
        }
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
        // let mut m = Matrix4x4::I;
        // let v = shift.0;
        // // m.0 = shuffle!(
        // //     m.0,
        // //     f32x16::splat(shift.0),
        // //     [0, 1, 2, 16, 4, 5, 6, 17, 8, 9, 10, 18, 12, 13, 14, 19]
        // // );
        // m.0.replace(3, v.extract(0));
        // m.0.replace(7, v.extract(1));
        // m.0.replace(11, v.extract(2));
        // m.0.replace(15, v.extract(3));
        Transform3::new_from_matrix(nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
            shift.x(),
            shift.y(),
            shift.z(),
        )))
    }

    pub fn scale(scale: Vec3) -> Self {
        Transform3::new_from_matrix(nalgebra::Matrix4::new_nonuniform_scaling(
            &nalgebra::Vector3::new(scale.x(), scale.y(), scale.z()),
        ))
    }

    pub fn axis_angle(axis: Vec3, radians: f32) -> Self {
        let axisangle = radians * nalgebra::Vector3::new(axis.x(), axis.y(), axis.z());

        let affine = nalgebra::Matrix4::from_scaled_axis(axisangle);
        Transform3::new_from_matrix(affine)
    }

    // pub fn rotation(quaternion: f32x4) -> Self {
    //     let quat = nalgebra::Quaternion::new()

    //     let affine = nalgebra::Matrix4::from_scaled_axis(axisangle);
    //     Transform3::new_from_matrix(affine)
    // }

    pub fn new_from_raw(forward: Matrix4x4, reverse: Matrix4x4) -> Self {
        Transform3 { forward, reverse }
    }
}

impl From<nalgebra::Matrix4<f32>> for Matrix4x4 {
    fn from(matrix: nalgebra::Matrix4<f32>) -> Self {
        // let slice: &[f32] = matrix.as_slice().into();
        let vec: Vec<f32> = matrix.as_slice().to_owned();
        let mut elements: f32x16 = f32x16::splat(0.0);
        for (i, v) in vec.iter().enumerate() {
            elements = elements.replace(i, *v);
        }
        Matrix4x4(elements)
    }
}

// impl From<Matrix4x4> for nalgebra::Matrix4<f32> {
//     fn from(matrix: Matrix4x4) -> Self {
//         unimplemented!()
//     }
// }

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
        Transform3::new_from_raw(rhs.reverse * self.reverse, rhs.forward * self.forward)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
