mod bdpt;
mod lt;
mod pt;
mod sppm;
pub mod utils;
pub use crate::camera::{Camera, CameraId};
use crate::config::RenderSettings;
use crate::math::*;
use crate::profile::Profile;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::world::World;

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;
pub use sppm::SPPMIntegrator;

use std::hash::Hash;
use std::sync::Arc;

// pub type CameraId = u8;

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum IntegratorType {
    PathTracing,
    LightTracing,
    BDPT,
    SPPM,
    MLT,
}

impl IntegratorType {
    pub fn from_string(string: &str) -> Self {
        match string {
            "PT" => IntegratorType::PathTracing,
            "LT" => IntegratorType::LightTracing,
            "BDPT" => IntegratorType::BDPT,
            "MLT" => IntegratorType::MLT,
            "SPPM" => IntegratorType::SPPM,
            _ => IntegratorType::PathTracing,
        }
    }
}

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    LightTracing(LightTracingIntegrator),
    BDPT(BDPTIntegrator),
    SPPM(SPPMIntegrator),
}

impl Integrator {
    pub fn from_settings_and_world(
        world: Arc<World>,
        integrator_type: IntegratorType,
        _cameras: &Vec<Camera>,
        settings: &RenderSettings,
    ) -> Option<Self> {
        let (lower, upper) = settings
            .wavelength_bounds
            .unwrap_or((VISIBLE_RANGE.lower, VISIBLE_RANGE.upper));
        assert!(lower < upper);
        match integrator_type {
            _ => Some(Integrator::PathTracing(PathTracingIntegrator {
                min_bounces: settings.min_bounces.unwrap_or(4),
                max_bounces: settings.max_bounces.unwrap(),
                world,
                russian_roulette: settings.russian_roulette.unwrap_or(true),
                light_samples: settings.light_samples.unwrap_or(4),
                only_direct: settings.only_direct.unwrap_or(false),
                wavelength_bounds: Bounds1D::new(lower, upper),
            })),
        }
    }
}

pub trait SamplerIntegrator: Sync + Send {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        camera_sample: ((f32, f32), CameraId),
        sample_id: usize,
        profile: &mut Profile,
    ) -> SingleWavelength;
}

pub enum Sample {
    ImageSample(SingleWavelength, (f32, f32)),
    LightSample(SingleWavelength, (f32, f32)),
}

#[allow(unused)]
pub trait GenericIntegrator: Send + Sync {
    fn preprocess(
        &mut self,
        sampler: &mut Box<dyn Sampler>,
        settings: &Vec<RenderSettings>,
        profile: &mut Profile,
    ) {
    }
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        settings: &RenderSettings,
        camera_sample: ((f32, f32), CameraId),
        sample_id: usize,
        samples: &mut Vec<(Sample, CameraId)>,
        profile: &mut Profile,
    ) -> SingleWavelength;
}
