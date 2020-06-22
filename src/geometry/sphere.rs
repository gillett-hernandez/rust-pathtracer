use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
    pub material_id: Option<MaterialId>,
    pub instance_id: usize,
}

impl Sphere {
    pub fn new(
        radius: f32,
        origin: Point3,
        material_id: Option<MaterialId>,
        instance_id: usize,
    ) -> Sphere {
        Sphere {
            radius,
            origin,
            material_id,
            instance_id,
        }
    }

    // fn solid_angle(&self, point: Point3, wi: Vec3) -> f32 {
    //     let cos_theta_max =
    //         (1.0 - self.radius * self.radius / (self.origin - point).norm_squared()).sqrt();
    //     2.0 * PI * (1.0 - cos_theta_max)
    // }
}

impl HasBoundingBox for Sphere {
    fn bounding_box(&self) -> AABB {
        AABB::new(
            self.origin - Vec3::from(self.radius),
            self.origin + Vec3::from(self.radius),
        )
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
            // time = r.time + (-b - discriminant_sqrt) / a;
            time = (-b - discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord::new(
                    time,
                    point,
                    (0.0, 0.0),
                    0.0,
                    normal,
                    self.material_id,
                    self.instance_id,
                ));
            }
            // time = r.time + (-b + discriminant_sqrt) / a;
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord::new(
                    time,
                    point,
                    (0.0, 0.0),
                    0.0,
                    normal,
                    self.material_id,
                    self.instance_id,
                ));
            }
        }
        None
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, f32) {
        let normal = random_on_unit_sphere(s.draw_2d());
        let point_on_sphere = self.origin + self.radius * normal;
        let direction = point_on_sphere - from;
        debug_assert!(direction.0.is_finite().all(), "{:?} {:?}", point_on_sphere, from);
        let pdf = direction.norm_squared()
            / ((normal * direction.normalized()).abs() * self.radius * self.radius * 4.0 * PI);
        if !pdf.is_finite() {
            println!("pdf was {:?}, direction: {:?}, normal: {:?}", pdf, direction, normal);

            (direction.normalized(), 0.0)
        } else {
            (direction.normalized(), pdf)
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> f32 {
        let direction = (to - from);
        let distance_squared = direction.norm_squared();
        let pdf = distance_squared
            / ((normal * direction.normalized()) * self.radius * self.radius * 4.0 * PI);
        debug_assert!(pdf.is_finite());
        pdf
    }
    fn get_instance_id(&self) -> usize {
        self.instance_id
    }
}
