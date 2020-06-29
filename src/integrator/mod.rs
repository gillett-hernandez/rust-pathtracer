mod bdpt;
mod lt;
mod pt;
pub use crate::camera::{Camera, CameraId};
use crate::config::RenderSettings;
use crate::math::*;
use crate::world::World;
use crate::INTERSECTION_TIME_OFFSET;

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;

use std::hash::Hash;
use std::sync::Arc;

// pub type CameraId = u8;

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum IntegratorType {
    PathTracing,
    LightTracing,
    BDPT,
    MLT,
}

impl IntegratorType {
    pub fn from_string(string: &str) -> Self {
        match string {
            "PT" => IntegratorType::PathTracing,
            "LT" => IntegratorType::LightTracing,
            "BDPT" => IntegratorType::BDPT,
            "MLT" => IntegratorType::MLT,
            _ => IntegratorType::PathTracing,
        }
    }
}

pub fn veach_v(world: &Arc<World>, point0: Point3, point1: Point3) -> bool {
    // returns if the points are visible
    let diff = point1 - point0;
    let norm = diff.norm();
    let tmax = norm * 0.99;
    let point0_to_point1 = Ray::new_with_time_and_tmax(point0, diff / norm, 0.0, tmax);
    let hit = world.hit(point0_to_point1, INTERSECTION_TIME_OFFSET, tmax);
    // if (point0.x() == 1.0 || point1.x() == 1.0) && !hit.as_ref().is_none() {
    //     // from back wall to something
    //     println!(
    //         "{:?} {:?}, hit was {:?}",
    //         point0,
    //         point1,
    //         hit.as_ref().unwrap()
    //     );
    // }
    hit.is_none()
}

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    LightTracing(LightTracingIntegrator),
    BDPT(BDPTIntegrator),
}

impl Integrator {
    pub fn from_settings_and_world(
        world: Arc<World>,
        integrator_type: IntegratorType,
        _cameras: &Vec<Camera>,
        settings: &RenderSettings,
    ) -> Option<Self> {
        match integrator_type {
            _ => Some(Integrator::PathTracing(PathTracingIntegrator {
                max_bounces: settings.max_bounces.unwrap(),
                world,
                russian_roulette: settings.russian_roulette.unwrap_or(true),
                light_samples: settings.light_samples.unwrap_or(4),
                only_direct: settings.only_direct.unwrap_or(false),
            })),
        }
    }
}

pub trait SamplerIntegrator: Sync + Send {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength;
}

pub enum Sample {
    ImageSample(SingleWavelength, (f32, f32)),
    LightSample(SingleWavelength, (f32, f32)),
}

pub trait GenericIntegrator: Send + Sync {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        settings: &RenderSettings,
        camera_sample: (Ray, CameraId),
        samples: &mut Vec<(Sample, CameraId)>,
    ) -> SingleWavelength;
}
