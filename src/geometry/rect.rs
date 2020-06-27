use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

fn vec_shuffle(vec: Vec3, axis: &Axis) -> Vec3 {
    match axis {
        Axis::X => Vec3::from_raw(shuffle!(vec.0, [2, 1, 0, 3])),
        Axis::Y => Vec3::from_raw(shuffle!(vec.0, [0, 2, 1, 3])),
        Axis::Z => vec,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AARect {
    pub size: (f32, f32),
    half_size: (f32, f32),
    pub normal: Axis,
    pub two_sided: bool,
    pub origin: Point3,
    pub material_id: MaterialId,
    pub instance_id: usize,
}

impl AARect {
    pub fn new(
        size: (f32, f32),
        origin: Point3,
        normal: Axis,
        two_sided: bool,
        material_id: MaterialId,
        instance_id: usize,
    ) -> Self {
        AARect {
            size,
            half_size: (size.0 / 2.0, size.1 / 2.0),
            origin,
            two_sided,
            normal,
            material_id,
            instance_id,
        }
    }
    pub fn from_quad(
        origin: Point3,
        normal: Axis,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        two_sided: bool,
        material_id: MaterialId,
        instance_id: usize,
    ) -> Self {
        let size = (x1 - x0, y1 - y0);
        let midpoint = (x0 + size.0 / 2.0, y0 + size.1 / 2.0);
        let offset = vec_shuffle(Vec3::new(midpoint.0, midpoint.1, 0.0), &normal);
        AARect {
            size,
            half_size: (size.0 / 2.0, size.1 / 2.0),
            origin: origin + offset,
            two_sided,
            normal,
            material_id,
            instance_id,
        }
    }
}

impl HasBoundingBox for AARect {
    fn bounding_box(&self) -> AABB {
        let v = vec_shuffle(
            Vec3::new(self.size.0 / 2.0, self.size.1 / 2.0, 0.0001),
            &self.normal,
        );
        AABB::new(self.origin - v, self.origin + v)
    }
}

impl Hittable for AARect {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let tmp_o = vec_shuffle(r.origin - self.origin, &self.normal);
        let tmp_d = vec_shuffle(r.direction, &self.normal);
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
        if xh < -self.half_size.0
            || xh > self.half_size.0
            || yh < -self.half_size.1
            || yh > self.half_size.1
        {
            return None;
        }
        let mut hit_normal = Vec3::from_axis(self.normal);
        if r.direction * hit_normal > 0.0 && self.two_sided {
            hit_normal = -hit_normal;
        }
        Some(HitRecord::new(
            t,
            r.point_at_parameter(t),
            (
                (xh - self.half_size.0) / self.size.0,
                (yh - self.half_size.1) / self.size.1,
            ),
            0.0,
            hit_normal,
            self.material_id,
            self.instance_id,
        ))
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let Sample2D { mut x, y } = s;
        let mut normal = Vec3::from_axis(self.normal);
        if self.two_sided {
            let choice = Sample1D { x }.choose(0.5, -1.0f32, 1.0f32);
            x = choice.0.x;
            normal = normal * choice.1;
        }
        let point = self.origin
            + vec_shuffle(
                Vec3::new((x - 0.5) * self.size.0, (y - 0.5) * self.size.1, 0.0),
                &self.normal,
            );
        let area = self.size.0 * self.size.1;
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
        let pdf = area_pdf * direction.norm_squared() / cos_i.abs();
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
        let area = self.size.0 * self.size.1;
        let distance_squared = direction.norm_squared();
        // TODO: affirm that it's fine to return 0.0 when not two sided.
        PDF::from(distance_squared / (cos_i.abs() * area))
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
