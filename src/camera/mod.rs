mod projective_camera;
mod realistic_camera;

pub use projective_camera::ProjectiveCamera;
pub use realistic_camera::RealisticCamera;

use crate::geometry::*;
use crate::math::*;

use std::marker::{Send, Sync};

pub type CameraId = usize;

macro_rules! generate_camera {
    ($s: ident, $($e:ident),+) => {
        #[derive(Debug, Clone)]
        pub enum $s {
            $(
                $e($e),
            )+
        }
        impl $s {
            pub fn get_ray(
                &self,
                sampler: &mut Box<dyn Sampler>,
                lambda: f32,
                s: f32,
                t: f32,
            ) -> (Ray, f32) {
                match self {
                    $(
                        $s::$e(inner) => inner.get_ray(sampler, lambda, s, t),
                    )+
                }
            }
            pub fn with_aspect_ratio(&self, aspect_ratio: f32) -> Self {
                match self {
                    $(
                        $s::$e(inner) => $s::$e(inner.clone().with_aspect_ratio(aspect_ratio)),
                    )+
                }
            }
            pub fn get_surface(&self) -> Option<&Instance> {
                match self {
                    $(
                        $s::$e(inner) => inner.get_surface(),
                    )+
                }
            }
            pub fn get_pixel_for_ray(&self, ray: Ray, lambda: f32) -> Option<(f32, f32)> {
                match self {
                    $(
                        $s::$e(inner) => inner.get_pixel_for_ray(ray, lambda),
                    )+
                }
            }

            pub fn eval_we(&self, _lambda: f32, normal: Vec3, from: Point3, to: Point3) -> (f32, PDF) {
                // TODO: once an accelerator for reverse tracing is implemented for realistic camera
                // or a "reverse trace" for the thin lens approximation is implemented for SimpleCamera,
                // use that to evaluate this
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
                mut sampler: &mut Box<dyn Sampler>,
                lambda: f32,
            ) -> (Ray, Vec3, PDF) {
                match self {
                    $(
                        $s::$e(inner) => {
                            let (ray, tau) = inner.get_ray(&mut sampler, lambda, film_sample.x, film_sample.y);
                            (ray, inner.direction, PDF::from(tau))
                        },
                    )+
                }

            }
        }

        unsafe impl Send for $s {}
        unsafe impl Sync for $s {}

    };
}

generate_camera! {Camera, ProjectiveCamera, RealisticCamera}
