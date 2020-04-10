use crate::math::*;
use std::ops::Mul;

pub struct Transform3 {
    matrix: [f32; 16],
}

impl Mul<Transform3> for Transform3 {
    type Output = Transform3;
    fn mul(&self, other: Transform3) -> Transform3 {
        other
    }
}

impl Mul<Vec3> for Transform3 {
    type Output = Vec3;
    fn mul(&self, other: Vec3) -> Vec3 {
        other
    }
}

impl Mul<Point3> for Transform3 {
    type Output = Point3;
    fn mul(&self, other: Point3) -> Point3 {
        other
    }
}

pub trait Transformable {
    fn transform_in_place(&mut self, transform: Transform3);
    fn transform(mut self, transform: Transform3) -> Self;
}
