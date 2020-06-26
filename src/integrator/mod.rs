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

use std::sync::Arc;

pub enum IntegratorType {
    PathTracing,
    LightTracing,
    BDPT,
    MLT,
}

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    LightTracing(LightTracingIntegrator),
    BDPT(BDPTIntegrator),
}

impl Integrator {
    pub fn new(
        world: Arc<World>,
        integrator_type: IntegratorType,
        cameras: Vec<Box<dyn Camera>>,
        settings: &RenderSettings,
    ) -> Self {
        match integrator_type {
            _ => Integrator::PathTracing(PathTracingIntegrator {
                max_bounces: 10,
                world,
                russian_roulette: true,
                light_samples: 4,
                only_direct: false,
            }),
            IntegratorType::BDPT => Integrator::BDPT(BDPTIntegrator {
                max_bounces: 10,
                world,
                specific_pair: None,
                cameras,
                camera_id: settings.camera_id.unwrap() as usize,
            }),
            IntegratorType::LightTracing => Integrator::LightTracing(LightTracingIntegrator {
                max_bounces: 10,
                world,
                russian_roulette: true,
                cameras,
            }),
        }
    }
}

impl From<&str> for IntegratorType {
    fn from(string: &str) -> Self {
        match string {
            "PT" => IntegratorType::PathTracing,
            "LT" => IntegratorType::LightTracing,
            "BDPT" => IntegratorType::BDPT,
            "MLT" => IntegratorType::MLT,
            _ => IntegratorType::PathTracing,
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
    fn color(&self, sampler: &mut Box<dyn Sampler>, samples: &mut Vec<(Sample, CameraId)>);
}
