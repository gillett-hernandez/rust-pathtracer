use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

pub struct Disk {
    pub radius: f32,
    pub normal: Vec3,
    pub two_sided: bool,
    pub origin: Point3,
    pub material_id: MaterialId,
    pub instance_id: usize,
    frame: TangentFrame,
}

impl Disk {
    pub fn new(
        radius: f32,
        origin: Point3,
        normal: Vec3,
        two_sided: bool,
        material_id: MaterialId,
        instance_id: usize,
    ) -> Self {
        Disk {
            radius,
            origin,
            two_sided,
            normal,
            material_id,
            instance_id,
            frame: TangentFrame::from_normal(normal),
        }
    }
}

impl HasBoundingBox for Disk {
    fn bounding_box(&self) -> AABB {
        let v = Vec3::new(self.radius / 2.0, self.radius / 2.0, 0.0001);
        let v = self.frame.to_world(&v);
        AABB::new(self.origin - v, self.origin + v)
    }
}

impl Hittable for Disk {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let tmp_o = self.frame.to_local(&Vec3::from(r.origin));
        let tmp_d = self.frame.to_local(&r.direction);
        if tmp_d.z() == 0.0 {
            return None;
        }
        let t = (-tmp_o.z()) / tmp_d.z();
        assert!(t.is_finite(), "{:?} {:?}", tmp_o, tmp_d);
        if t < t0 || t > t1 || t >= r.tmax {
            return None;
        }
        let xh = tmp_o.x() + t * tmp_d.x();
        let yh = tmp_o.y() + t * tmp_d.y();
        if xh * xh + yh * yh > self.radius * self.radius {
            return None;
        }
        let mut hit_normal = self.normal;
        if r.direction * hit_normal > 0.0 && self.two_sided {
            hit_normal = -hit_normal;
        }
        Some(HitRecord::new(
            t,
            r.point_at_parameter(t),
            (0.0, 0.0),
            0.0,
            hit_normal,
            self.material_id,
            self.instance_id,
        ))
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let Sample2D { x, y } = s;
        let point = self.origin
            + vec_shuffle(
                Vec3::new((x - 0.5) * self.size.0, (y - 0.5) * self.size.1, 0.0),
                &self.normal,
            );
        let area = self.size.0 * self.size.1;
        let normal = Vec3::from_axis(self.normal);
        (point, normal, (1.0 / area).into())
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, PDF) {
        let (point, normal, area_pdf) = self.sample_surface(s.draw_2d());
        let direction = point - from;
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
        let area = self.size.0 * self.size.1;
        let distance_squared = direction.norm_squared();
        PDF::from(distance_squared / ((normal * direction.normalized()).abs() * area))
    }

    fn surface_area(&self, transform: &Transform3) -> f32 {
        let transformed_axes = transform.axis_transform();
        let transform_multiplier = match self.normal {
            Axis::X => transformed_axes.1.norm() * transformed_axes.2.norm(),
            Axis::Y => transformed_axes.0.norm() * transformed_axes.2.norm(),
            Axis::Z => transformed_axes.0.norm() * transformed_axes.1.norm(),
        };
        transform_multiplier * self.size.0 * self.size.1
    }
    fn get_instance_id(&self) -> usize {
        self.instance_id
    }
    fn get_material_id(&self) -> MaterialId {
        self.material_id
    }
}
