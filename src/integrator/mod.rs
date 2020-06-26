mod bdpt;
mod lt;
mod pt;
use crate::camera::{Camera, CameraId};
use crate::config::RenderSettings;
use crate::math::*;
use crate::world::World;

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;

use std::hash::Hash;
use std::sync::Arc;

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

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    LightTracing(LightTracingIntegrator),
    BDPT(BDPTIntegrator),
}

impl Integrator {
    pub fn from_settings_and_world(
        world: Arc<World>,
        integrator_type: IntegratorType,
        cameras: &Vec<Camera>,
        settings: &RenderSettings,
    ) -> Option<Self> {
        match integrator_type {
            _ => Some(Integrator::PathTracing(PathTracingIntegrator {
                max_bounces: 10,
                world,
                russian_roulette: true,
                light_samples: 4,
                only_direct: false,
            })),
            _ => None,
        }
    }
}

pub trait SamplerIntegrator: Sync + Send {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength;
}

pub enum Sample {
    ImageSample(SingleWavelength, (usize, usize)),
    LightSample(SingleWavelength, (usize, usize)),
}

pub trait GenericIntegrator: Send + Sync {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        camera_ray: Ray,
        samples: &mut Vec<(Sample, CameraId)>,
    ) -> SingleWavelength;
}
