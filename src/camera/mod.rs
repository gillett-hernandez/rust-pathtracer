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
pub trait Camera<L: Field, E: Field> {
    // determine how sample_we is different from get_ray
    fn get_ray(&self, sampler: &mut Box<dyn Sampler>, lambda: L, u: f32, v: f32) -> (Ray, f32);
    fn with_aspect_ratio(self, aspect_ratio: f32) -> Self;
    fn get_surface(&self) -> Option<&Instance> {
        None
    }
    fn get_pixel_for_ray(&self, ray: Ray, lambda: f32) -> Option<(f32, f32)>;
    // the W_e function has units SW^-1 where W is Watts and S is some unit of sensor response (voltage, current, change in film density, or deflector of a meter needle)
    // defined as dS(x, w, lambda) / dPhi(x, w, lambda) where x is a position on the sensor(film), w is the incoming direction
    // for physical sensors, W_e is the spectral flux responsivity, or sensor response per unit power arriving at x from direction w
    // for hypothetical simulated sensors, typically W_e is the exitant importance function, and has units W^-1, as the S is assumed to be dimensionless
    // typically W_e is assumed to be the response of a linear sensor, i.e. 2x incoming power is 2x sensor response.
    // pixel filtering could go here.
    // nonlinear sensors are typically modeled by applying a nonlinear response function after the linear response data has been collected
    // TODO: refactor this to actually coorespond to the function as defined by Veach (x == position on film, w = incoming direction, l = lambda)
    // TODO: after the above has been completed, address the fact that if lambda is any nonscalar value (f32x4, etc), then eval_we would return 0 for all lanes/channels except for the main wavelength
    // perhaps by splitting the WavelengthEnergy packet into its constituents, and sampling paths to the film for each lambda
    fn eval_we(
        &self,
        lambda: L,
        normal: Vec3,
        from: Point3,
        to: Point3,
    ) -> (E, PDF<E, SolidAngle>);
    fn sample_we(
        &self,
        film_sample: Sample2D,
        sampler: &mut Box<dyn Sampler>,
        lambda: L,
    ) -> (Ray, Vec3, PDF<E, SolidAngle>);
    fn sample_surface(&self, lambda: f32, from: Point3) -> Option<(Ray, f32, PDF<E, SolidAngle>)> {
        None
    }
}

macro_rules! generate_camera {

    ($name: ident, $l: ty, $e: ty, $($item:ident),+) => {

        impl Camera<$l, $e> for $name {
             fn get_ray(
                &self,
                sampler: &mut Box<dyn Sampler>,
                lambda: f32,
                s: f32,
                t: f32,
            ) -> (Ray, f32) {
                match self {
                    $(
                        $name::$item(inner) => inner.get_ray(sampler, lambda, s, t),
                    )+
                }
            }
             fn with_aspect_ratio(self, aspect_ratio: f32) -> Self {
                match self {
                    $(
                        $name::$item(inner) => $name::$item(inner.clone().with_aspect_ratio(aspect_ratio)),
                    )+
                }
            }
             fn get_surface(&self) -> Option<&Instance> {
                match self {
                    $(
                        $name::$item(inner) => inner.get_surface(),
                    )+
                }
            }
             fn get_pixel_for_ray(&self, ray: Ray, lambda: f32) -> Option<(f32, f32)> {
                match self {
                    $(
                        $name::$item(inner) => inner.get_pixel_for_ray(ray, lambda),
                    )+
                }
            }

             fn eval_we(&self, _lambda: f32, normal: Vec3, from: Point3, to: Point3) -> (f32, PDF<$e, SolidAngle>) {
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
            ) -> (Ray, Vec3, PDF<$e, SolidAngle>) {
                match self {
                    $(
                        $name::$item(inner) => inner.sample_we(film_sample, sampler, lambda),
                    )+
                }

            }
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

    };
    ($name: ident, $($item:ident),+) => {
        #[derive(Debug, Clone)]
        pub enum $name {
            $(
                $item($item),
            )+
        }
    };
}

#[cfg(not(feature = "realistic_camera"))]
generate_camera! {CameraEnum, ProjectiveCamera, PanoramaCamera}
#[cfg(not(feature = "realistic_camera"))]
generate_camera! {CameraEnum, f32, f32, ProjectiveCamera, PanoramaCamera}



#[cfg(feature = "realistic_camera")]
generate_camera! {CameraEnum, ProjectiveCamera, PanoramaCamera, RealisticCamera}
#[cfg(feature = "realistic_camera")]
generate_camera! {CameraEnum, f32, f32, ProjectiveCamera, PanoramaCamera, RealisticCamera}
