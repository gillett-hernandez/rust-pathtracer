use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};
use crate::math::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
}

impl Sphere {
    pub fn new(radius: f32, origin: Point3) -> Sphere {
        Sphere { radius, origin }
    }

    // fn solid_angle(&self, point: Point3, wi: Vec3) -> f32 {
    //     let cos_theta_max =
    //         (1.0 - self.radius * self.radius / (self.origin - point).norm_squared()).sqrt();
    //     2.0 * PI * (1.0 - cos_theta_max)
    // }
}

impl HasBoundingBox for Sphere {
    fn aabb(&self) -> AABB {
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
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(HitRecord::new(
                    time,
                    point,
                    (0.0, 0.0),
                    0.0,
                    normal,
                    0.into(),
                    0,
                    None,
                ));
            }
            // time = r.time + (-b + discriminant_sqrt) / a;
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(HitRecord::new(
                    time,
                    point,
                    (0.0, 0.0),
                    0.0,
                    normal,
                    0.into(),
                    0,
                    None,
                ));
            }
        }
        None
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        let normal = random_on_unit_sphere(s);
        let point_on_sphere = self.origin + self.radius * normal;
        let surface_area = self.radius * self.radius * 4.0 * PI;
        (point_on_sphere, normal, PDF::from(1.0 / surface_area))
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        let (point_on_sphere, normal, area_pdf) = self.sample_surface(s);
        let direction = point_on_sphere - from;
        debug_assert!(
            direction.0.is_finite().all(),
            "{:?} {:?}",
            point_on_sphere,
            from
        );
        let pdf = area_pdf * direction.norm_squared() / (normal * direction.normalized()).abs();
        if !pdf.0.is_finite() {
            println!(
                "pdf was {:?}, direction: {:?}, normal: {:?}",
                pdf, direction, normal
            );

            (direction.normalized(), 0.0.into())
        } else {
            (direction.normalized(), pdf)
        }
    }
    fn psa_pdf(&self, cos_o: f32, from: Point3, to: Point3) -> PDF {
        let direction = to - from;
        let distance_squared = direction.norm_squared();
        let pdf = distance_squared / (cos_o * self.radius * self.radius * 4.0 * PI);
        debug_assert!(pdf.is_finite() && pdf >= 0.0);
        pdf.into()
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        let transformed_axes = transform.axis_transform();
        self.radius
            * self.radius
            * 4.0
            * PI
            * transformed_axes.0.norm()
            * transformed_axes.1.norm()
            * transformed_axes.2.norm()
    }
}
