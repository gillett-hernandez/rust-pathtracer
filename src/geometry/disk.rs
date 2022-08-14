use crate::prelude::*;

use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disk {
    pub radius: f32,
    pub two_sided: bool,
    pub origin: Point3,
}

impl Disk {
    pub fn new(radius: f32, origin: Point3, two_sided: bool) -> Self {
        Disk {
            radius,
            origin,
            two_sided,
        }
    }
}

impl HasBoundingBox for Disk {
    fn aabb(&self) -> AABB {
        let v = Vec3::new(self.radius / 2.0, self.radius / 2.0, 0.001);
        AABB::new(self.origin - v, self.origin + v)
    }
}

impl Hittable for Disk {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let tmp_o = r.origin - self.origin;
        let tmp_d = r.direction;
        if tmp_d.z() == 0.0 {
            return None;
        }
        let t = (-tmp_o.z()) / tmp_d.z();
        debug_assert!(t.is_finite(), "{:?} {:?}", tmp_o, tmp_d);
        if t <= t0 || t > t1 || t >= r.tmax {
            return None;
        }
        let xh = tmp_o.x() + t * tmp_d.x();
        let yh = tmp_o.y() + t * tmp_d.y();
        if xh * xh + yh * yh > self.radius * self.radius {
            return None;
        }
        let mut hit_normal = Vec3::Z;
        if r.direction * hit_normal > 0.0 && self.two_sided {
            hit_normal = -hit_normal;
        }
        Some(HitRecord::new(
            t,
            r.point_at_parameter(t),
            // TODO: compute UV for disk
            (0.0, 0.0),
            0.0,
            hit_normal,
            0.into(),
            0,
            None,
        ))
    }
    fn sample_surface(&self, mut s: Sample2D) -> (Point3, Vec3, PDF) {
        let mut normal = Vec3::Z;
        // if dual sided, randomly pick the opposite side when sampling
        if self.two_sided {
            let choice = Sample1D::new(s.x).choose(0.5, -1.0f32, 1.0f32);
            s.x = choice.0.x;
            normal = normal * choice.1;
        }
        // otherwise stick with Z+
        let point = self.origin + self.radius * random_in_unit_disk(s);
        let area = PI * self.radius * self.radius;
        (point, normal, (1.0 / area).into())
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        let (point, normal, area_pdf) = self.sample_surface(s);
        debug_assert!(point.0.is_finite().all());
        debug_assert!(normal.0.is_finite().all());
        debug_assert!(area_pdf.0.is_finite());
        let direction = point - from;
        let cos_i = normal * direction.normalized();

        let pdf = area_pdf * direction.norm_squared() / cos_i.abs();
        if !pdf.0.is_finite() {
            println!("pdf was inf, {:?}", direction);
            (direction.normalized(), 0.0.into())
        } else {
            (direction.normalized(), pdf)
        }
    }
    fn psa_pdf(&self, cos_o: f32, from: Point3, to: Point3) -> PDF {
        let direction = to - from;

        let area = PI * self.radius * self.radius;
        let distance_squared = direction.norm_squared();
        PDF::from(distance_squared / ((cos_o.abs() + 0.00001) * area))
    }

    fn surface_area(&self, transform: &Transform3) -> f32 {
        let transformed_axes = transform.axis_transform();
        transformed_axes.0.norm() * transformed_axes.1.norm() * PI * self.radius * self.radius
    }
}
