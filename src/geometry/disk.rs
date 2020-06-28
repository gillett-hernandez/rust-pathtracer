use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disk {
    pub radius: f32,
    pub two_sided: bool,
    pub origin: Point3,
    pub material_id: MaterialId,
    pub instance_id: usize,
}

impl Disk {
    pub fn new(
        radius: f32,
        origin: Point3,
        two_sided: bool,
        material_id: MaterialId,
        instance_id: usize,
    ) -> Self {
        Disk {
            radius,
            origin,
            two_sided,
            material_id,
            instance_id,
        }
    }
}

impl HasBoundingBox for Disk {
    fn bounding_box(&self) -> AABB {
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
        assert!(t.is_finite(), "{:?} {:?}", tmp_o, tmp_d);
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
            self.material_id,
            self.instance_id,
        ))
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let Sample2D { x: mut u, y: v } = s;
        let r = v.sqrt();
        let mut normal = Vec3::Z;
        if self.two_sided {
            let choice = Sample1D { x: u }.choose(0.5, -1.0f32, 1.0f32);
            u = choice.0.x;
            normal = normal * choice.1;
        }
        let phi = u * PI * 2.0;
        let (sin, cos) = phi.sin_cos();
        let point = self.origin + Vec3::new(self.radius * r * cos, self.radius * r * sin, 0.0);
        let area = PI * self.radius * self.radius;
        (point, normal, (1.0 / area).into())
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, PDF) {
        let (point, normal, area_pdf) = self.sample_surface(s.draw_2d());
        let direction = point - from;
        let cos_i = normal * direction.normalized();
        if !self.two_sided {
            if cos_i < 0.0 {
                return (direction.normalized(), 0.0.into());
            }
        }
        let pdf = area_pdf * direction.norm_squared() / ((normal * direction.normalized()).abs());
        if !pdf.0.is_finite() {
            // println!("pdf was inf, {:?}", direction);
            (direction.normalized(), 0.0.into())
        } else {
            (direction.normalized(), pdf)
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        let direction = to - from;
        let cos_i = normal * direction.normalized();
        if !self.two_sided {
            if cos_i < 0.0 {
                return 0.0.into();
            }
        }
        let area = PI * self.radius * self.radius;
        let distance_squared = direction.norm_squared();
        PDF::from(distance_squared / ((normal * direction.normalized()).abs() * area))
    }

    fn surface_area(&self, transform: &Transform3) -> f32 {
        let transformed_axes = transform.axis_transform();
        transformed_axes.0.norm() * transformed_axes.1.norm() * PI * self.radius * self.radius
    }
    fn get_instance_id(&self) -> usize {
        self.instance_id
    }
    fn get_material_id(&self) -> MaterialId {
        self.material_id
    }
}
