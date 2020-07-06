use crate::aabb::{HasBoundingBox, AABB};
use crate::geometry::mesh::Mesh;
use crate::hittable::{HitRecord, Hittable};
use crate::math::*;

use packed_simd::{f32x4, i32x4};

pub fn vec_shuffle(vec: f32x4, m: u32) -> f32x4 {
    match m {
        0 => shuffle!(vec, [0, 1, 2, 3]),
        1 => shuffle!(vec, [1, 0, 2, 3]),
        2 => shuffle!(vec, [2, 0, 1, 3]),
        _ => vec,
    }
}

#[derive(Clone, Copy)]
pub struct Triangle<'a> {
    pub idx: usize,
    pub vertices: &'a Vec<Point3>,
    pub v_indices: &'a Vec<usize>,
}

impl<'a> Triangle<'a> {
    pub fn new(idx: usize, vertices: &'a Vec<Point3>, v_indices: &'a Vec<usize>) -> Self {
        Triangle {
            idx,
            vertices,
            v_indices,
        }
    }
}

impl<'a> HasBoundingBox for Triangle<'a> {
    fn aabb(&self) -> AABB {
        let p0 = self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.v_indices[3 * self.idx + 2]];
        AABB::new(p0, p1).grow(&p2)
    }
}

impl<'a> Hittable for Triangle<'a> {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let p0 = self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.v_indices[3 * self.idx + 2]];
        let mut p0t = p0 - r.origin;
        let mut p1t = p1 - r.origin;
        let mut p2t = p2 - r.origin;
        let direction = r.direction.0;
        let max_axis_value = direction.max_element();
        let mask = direction.ge(f32x4::splat(max_axis_value));
        let max_axis = mask
            .select(i32x4::new(0, 1, 2, 3), i32x4::splat(0))
            .max_element() as u32;
        let kz = max_axis;
        let d: f32x4 = vec_shuffle(direction, kz);
        let [dx, dy, dz, _]: [f32; 4] = d.into();
        p0t.0 = vec_shuffle(p0t.0, kz);
        p1t.0 = vec_shuffle(p1t.0, kz);
        p2t.0 = vec_shuffle(p2t.0, kz);
        let [mut p0t_x, mut p0t_y, mut p0t_z, _]: [f32; 4] = p0t.0.into();
        let [mut p1t_x, mut p1t_y, mut p1t_z, _]: [f32; 4] = p1t.0.into();
        let [mut p2t_x, mut p2t_y, mut p2t_z, _]: [f32; 4] = p2t.0.into();
        let sx = -dx / dz;
        let sy = -dy / dz;
        let sz = 1.0 / dz;
        p0t_x += sx * p0t_z;
        p1t_x += sx * p1t_z;
        p2t_x += sx * p2t_z;
        p0t_y += sy * p0t_z;
        p1t_y += sy * p1t_z;
        p2t_y += sy * p2t_z;
        let mut e0 = p1t_x * p2t_y - p1t_y * p2t_x;
        let mut e1 = p2t_x * p0t_y - p2t_y * p0t_x;
        let mut e2 = p0t_x * p1t_y - p0t_y * p1t_x;
        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            let p2txp1ty: f64 = (p2t_x as f64) * (p1t_y as f64);
            let p2typ1tx: f64 = (p2t_y as f64) * (p1t_x as f64);
            e0 = (p2typ1tx - p2txp1ty) as f32;

            let p0txp2ty: f64 = (p0t_x as f64) * (p2t_y as f64);
            let p0typ2tx: f64 = (p0t_y as f64) * (p2t_x as f64);
            e1 = (p0typ2tx - p0txp2ty) as f32;

            let p1txp0ty: f64 = (p1t_x as f64) * (p0t_y as f64);
            let p1typ0tx: f64 = (p1t_y as f64) * (p0t_x as f64);
            e2 = (p1typ0tx - p1txp0ty) as f32;
        }

        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return None;
        }
        let det = e0 + e1 + e2;
        if det == 0.0 {
            return None;
        }

        p0t_z *= sz;
        p1t_z *= sz;
        p2t_z *= sz;

        let t_scaled = e0 * p0t_z + e1 * p1t_z + e2 * p2t_z;

        if (det < 0.0 && (t_scaled >= t0 * det || t_scaled < t1 * det))
            || (det > 0.0 && (t_scaled <= t0 * det || t_scaled > t1 * det))
        {
            return None;
        }

        let inv_det = det.recip();
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;

        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        Some(HitRecord::new(
            t_scaled * inv_det,
            Point3::from(b0 * Vec3::from(p0) + b1 * Vec3::from(p1) + b2 * Vec3::from(p2)),
            (0.0, 0.0),
            0.0,
            dp02.cross(dp12).normalized(),
            0.into(),
            0,
            None,
        ))
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        (Vec3::ZERO, 0.0.into())
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        (Point3::ORIGIN, Vec3::ZERO, 0.0.into())
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        0.0.into()
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        // calculates the surface area using heron's formula.
        let p0 = *transform * self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = *transform * self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = *transform * self.vertices[self.v_indices[3 * self.idx + 2]];
        let d02 = (p2 - p0).norm();
        let d01 = (p1 - p0).norm();
        let d12 = (p2 - p1).norm();
        let s = 0.5 * (d02 + d01 + d12);
        (s * (s - d01) * (s - d12) * (s - d02)).sqrt()
    }
}
