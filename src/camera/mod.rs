mod projective_camera;
// mod simple_camera;

pub use projective_camera::ProjectiveCamera;

use crate::geometry::*;
use crate::math::*;

use std::marker::{Send, Sync};

pub type CameraId = u8;

#[derive(Debug, Clone)]
pub enum Camera {
    ProjectiveCamera(ProjectiveCamera),
}

impl Camera {
    pub fn get_ray(&self, sample: Sample2D, s: f32, t: f32) -> Ray {
        match self {
            Camera::ProjectiveCamera(inner) => inner.get_ray(sample, s, t),
            // Camera::SimpleCamera(inner) => inner.get_ray(sample, s, t),
        }
    }
    pub fn with_aspect_ratio(&self, aspect_ratio: f32) -> Self {
        match self {
            Camera::ProjectiveCamera(inner) => {
                Camera::ProjectiveCamera(inner.clone().with_aspect_ratio(aspect_ratio))
            } // }
              // Camera::SimpleCamera(inner) => {
              // Camera::SimpleCamera(inner.with_aspect_ratio(aspect_ratio))
        }
    }
    pub fn get_surface(&self) -> Option<&Instance> {
        match self {
            Camera::ProjectiveCamera(inner) => inner.get_surface(),
            // Camera::SimpleCamera(inner) => inner.get_surface(),
        }
    }
    pub fn get_pixel_for_ray(&self, ray: Ray) -> Option<(f32, f32)> {
        match self {
            Camera::ProjectiveCamera(inner) => inner.get_pixel_for_ray(ray),
            // Camera::SimpleCamera(inner) => inner.get_pixel_for_ray(ray),
        }
    }

    pub fn eval_we(&self, normal: Vec3, from: Point3, to: Point3) -> (f32, PDF) {
        // from is on surface of camera
        // (0.01, 1.0.into())
        // (1.0, 0.01.into())
        let direction = to - from;
        if direction * normal < 0.0 {
            // connection from opposite side.
            (0.0, 0.0.into())
        } else {
            (1.0, 1.0.into())
        }
    }

    pub fn sample_we(
        &self,
        film_sample: Sample2D,
        aperture_sample: Sample2D,
        _lambda: f32,
    ) -> (Ray, Vec3, PDF) {
        // (Ray::new(Point3::ORIGIN, Vec3::Z), Vec3::Z, 0.0.into())
        match self {
            Camera::ProjectiveCamera(cam) => (
                cam.get_ray(aperture_sample, film_sample.x, film_sample.y),
                cam.direction,
                1.0.into(),
            ),
        }
    }
}

unsafe impl Send for Camera {}
unsafe impl Sync for Camera {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_debug() {
        let look_from = Point3::new(-5.0, 0.0, 0.0);
        let _look_at = Point3::ZERO;

        let origin = look_from;
        let lens_radius = 0.01;
        let u = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 0.0, -1.0);
        let w = Vec3::new(-1.0, 0.0, 0.0);
        let focus_dist = 5.0;
        let half_width = 0.5;
        let half_height = 0.3;
        let rd: Vec3 = lens_radius * random_in_unit_disk(Sample2D::new_random_sample());
        let offset = u * rd.x() + v * rd.y();
        let _ray_origin: Point3 = origin + offset;
        // println!("{:?}", ;
        let s = 0.3;
        let t = 0.7;
        let u_halfwidth_focus_dist = u * half_width * focus_dist;
        let v_halfheight_focus_dist = v * half_height * focus_dist;
        let _ray_direction = (u_halfwidth_focus_dist * (s * 2.0 - 1.0)
            + v_halfheight_focus_dist * (t * 2.0 - 1.0)
            - w * focus_dist
            - offset)
            .normalized();
        // let un_normalized = Vec3::new()
    }
}
