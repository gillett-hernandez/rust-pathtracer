mod panorama_camera;
mod projective_camera;
#[cfg(feature = "realistic_camera")]
mod realistic_camera;

pub use panorama_camera::PanoramaCamera;
pub use projective_camera::ProjectiveCamera;
#[cfg(feature = "realistic_camera")]
pub use realistic_camera::RealisticCamera;

use crate::geometry::*;
use crate::prelude::*;

use std::marker::{Send, Sync};

pub type CameraId = usize;

#[allow(unused_variables)]
pub trait Camera {
    // determine how sample_we is different from get_ray
    fn get_ray(&self, sampler: &mut Box<dyn Sampler>, lambda: f32, u: f32, v: f32) -> (Ray, f32);
    fn with_aspect_ratio(self, aspect_ratio: f32) -> Self;
    fn get_surface(&self) -> Option<&Instance> {
        None
    }
    fn get_pixel_for_ray(&self, ray: Ray, lambda: f32) -> Option<(f32, f32)>;
    fn eval_we(&self, lambda: f32, normal: Vec3, from: Point3, to: Point3) -> (f32, PDF);
    fn sample_we(
        &self,
        film_sample: Sample2D,
        sampler: &mut Box<dyn Sampler>,
        lambda: f32,
    ) -> (Ray, Vec3, PDF);
    fn sample_surface(&self, lambda: f32, from: Point3) -> Option<(Ray, f32, PDF)> {
        None
    }
}

macro_rules! generate_camera {
    ($s: ident, $($e:ident),+) => {
        #[derive(Debug, Clone)]
        pub enum $s {
            $(
                $e($e),
            )+
        }
        impl Camera for $s {
             fn get_ray(
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
             fn with_aspect_ratio(self, aspect_ratio: f32) -> Self {
                match self {
                    $(
                        $s::$e(inner) => $s::$e(inner.clone().with_aspect_ratio(aspect_ratio)),
                    )+
                }
            }
             fn get_surface(&self) -> Option<&Instance> {
                match self {
                    $(
                        $s::$e(inner) => inner.get_surface(),
                    )+
                }
            }
             fn get_pixel_for_ray(&self, ray: Ray, lambda: f32) -> Option<(f32, f32)> {
                match self {
                    $(
                        $s::$e(inner) => inner.get_pixel_for_ray(ray, lambda),
                    )+
                }
            }

             fn eval_we(&self, _lambda: f32, normal: Vec3, from: Point3, to: Point3) -> (f32, PDF) {
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

             fn sample_we(
                &self,
                film_sample: Sample2D,
                sampler: &mut Box<dyn Sampler>,
                lambda: f32,
            ) -> (Ray, Vec3, PDF) {
                match self {
                    $(
                        $s::$e(inner) => inner.sample_we(film_sample, sampler, lambda)
                        ,
                    )+
                }

            }
        }

        unsafe impl Send for $s {}
        unsafe impl Sync for $s {}

    };
}

#[cfg(not(feature = "realistic_camera"))]
generate_camera! {CameraEnum, ProjectiveCamera, PanoramaCamera}
#[cfg(feature = "realistic_camera")]
generate_camera! {CameraEnum, ProjectiveCamera, PanoramaCamera, RealisticCamera}
