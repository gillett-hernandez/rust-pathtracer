use crate::prelude::*;

use crate::aabb::{HasBoundingBox, AABB};
use crate::hittable::{HitRecord, Hittable};

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
            let mut time: f32 = (-b - discriminant_sqrt) / a;
            // trace!(
            //     "tracing sphere, time = {}, needs to be in [{}, {}]",
            //     time,
            //     t0,
            //     t1.min(r.tmax)
            // );
            let point: Point3;
            let normal: Vec3;

            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(HitRecord::new(
                    time,
                    point,
                    UV(0.0, 0.0),
                    0.0,
                    normal,
                    0.into(),
                    0,
                    None,
                ));
            }
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(HitRecord::new(
                    time,
                    point,
                    UV(0.0, 0.0),
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
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF<f32, Area>) {
        let normal = random_on_unit_sphere(s);
        let point_on_sphere = self.origin + self.radius * normal;
        let surface_area = self.radius * self.radius * 4.0 * PI;

        (point_on_sphere, normal, PDF::from(1.0 / surface_area))
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF<f32, SolidAngle>) {
        // TODO: replace this with perfect hemisphere sampling
        // i.e. https://schuttejoe.github.io/post/arealightsampling/
        // or https://momentsingraphics.de/Media/I3D2019/Peters2019-SamplingSphericalCaps.pdf

        let (point_on_sphere, normal, area_pdf) = self.sample_surface(s);
        let direction = point_on_sphere - from;
        debug_assert!(
            direction.0.is_finite().all(),
            "{:?} {:?}",
            point_on_sphere,
            from
        );

        let normal_dot_direction = (normal * direction.normalized()).abs();
        // Coerce pdf tag. note that this is not the projected solid angle pdf
        // because the normal dot direction is wrt the surface of the light at the sampled point,
        // rather than the surface at point `from`
        let pdf: PDF<f32, SolidAngle> =
            PDF::new(*area_pdf * direction.norm_squared() / normal_dot_direction);
        if !(*pdf).is_finite() {
            lazy_static! {
                static ref LOGGED_CELL: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
            }
            if !LOGGED_CELL.fetch_or(true, std::sync::atomic::Ordering::AcqRel) {
                warn!(
                    "pdf was inf, {:?}, area_pdf: {:?}, n * d: {:?}",
                    pdf, area_pdf, normal_dot_direction,
                );
            }

            (direction.normalized(), 0.0.into())
        } else {
            (direction.normalized(), pdf)
        }
    }
    fn psa_pdf(
        &self,
        cos_o: f32,
        cos_i: f32,
        from: Point3,
        to: Point3,
    ) -> PDF<f32, ProjectedSolidAngle> {
        let direction = to - from;
        let distance_squared = direction.norm_squared();
        let area_pdf: PDF<_, Area> = (self.radius * self.radius * 4.0 * PI).recip().into();
        // let pdf = distance_squared / (cos_o.abs() * self.radius * self.radius * 4.0 * PI);
        let pdf = area_pdf.convert_to_projected_solid_angle(cos_i, cos_o, distance_squared);
        debug_assert!(
            (*pdf).is_finite() && *pdf >= 0.0,
            "{:?} {:?} {:?}",
            distance_squared,
            cos_o,
            self.radius
        );
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
