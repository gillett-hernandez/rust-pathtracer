use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
    pub material_id: Option<MaterialId>,
}

impl Sphere {
    pub fn new(radius: f32, origin: Point3, material_id: Option<MaterialId>) -> Sphere {
        Sphere {
            radius,
            origin,
            material_id,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let oc: Vec3 = r.origin - self.origin;
        let a = r.direction * r.direction;
        let b = oc * r.direction;
        let c = oc * oc - self.radius * self.radius;
        let discriminant = b * b - a * c;
        let discriminant_sqrt = discriminant.sqrt();
        if discriminant > 0.0 {
            let mut time: f32;
            let point: Point3;
            let normal: Vec3;
            time = (-b - discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord::new(time, point, normal, self.material_id));
            }
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord::new(time, point, normal, self.material_id));
            }
        }
        None
    }
}

impl HasBoundingBox for Sphere {
    fn bounding_box(&self) -> AABB {
        AABB::new(
            self.origin - Vec3::new(self.radius, self.radius, self.radius),
            self.origin + Vec3::new(self.radius, self.radius, self.radius),
        )
    }
}
