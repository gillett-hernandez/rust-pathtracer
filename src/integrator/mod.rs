use crate::prelude::*;

mod bdpt;
mod lt;
mod pt;
// mod sppm;
pub mod utils;

use crate::parsing::config::{IntegratorKind, RenderSettings};

use crate::profile::Profile;
use crate::world::World;
use math::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;
// pub use sppm::SPPMIntegrator;

use std::hash::Hash;
use std::sync::Arc;

// pub type CameraId = u8;

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum IntegratorType {
    PathTracing,
    LightTracing,
    BDPT,
    // SPPM,
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

impl From<IntegratorKind> for IntegratorType {
    fn from(data: IntegratorKind) -> Self {
        match data {
            IntegratorKind::PT { .. } => IntegratorType::PathTracing,
            IntegratorKind::LT { .. } => IntegratorType::LightTracing,
            IntegratorKind::BDPT { .. } => IntegratorType::BDPT,
        }
    }
}

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    LightTracing(LightTracingIntegrator),
    BDPT(BDPTIntegrator),
    // SPPM(SPPMIntegrator),
}

impl Integrator {
    pub fn from_settings_and_world(
        world: Arc<World>,
        integrator_type: IntegratorType,
        _cameras: &[CameraEnum],
        settings: &RenderSettings,
    ) -> Option<Self> {
        let bounds = settings
            .wavelength_bounds
            .unwrap_or((VISIBLE_RANGE.lower, VISIBLE_RANGE.upper))
            .into();

        let max_bounces = settings.max_bounces.unwrap();
        let russian_roulette = settings.russian_roulette.unwrap_or(true);
        match (integrator_type, settings.integrator) {
            (IntegratorType::BDPT, IntegratorKind::BDPT { .. }) => {
                Some(Integrator::BDPT(BDPTIntegrator {
                    max_bounces,
                    world,
                    wavelength_bounds: bounds,
                }))
            }
            (IntegratorType::LightTracing, IntegratorKind::LT { camera_samples }) => {
                Some(Integrator::LightTracing(LightTracingIntegrator {
                    max_bounces,
                    world,
                    russian_roulette,
                    camera_samples,
                    wavelength_bounds: bounds,
                }))
            }
            (IntegratorType::PathTracing, IntegratorKind::PT { light_samples }) => {
                Some(Integrator::PathTracing(PathTracingIntegrator {
                    min_bounces: settings.min_bounces.unwrap_or(4),
                    max_bounces,
                    world,
                    russian_roulette,
                    light_samples,
                    only_direct: settings.only_direct.unwrap_or(false),
                    wavelength_bounds: bounds,
                }))
            }
            _ => {
                warn!("constructing pathtracing integrator as fallback, since IntegratorType did not match any supported integrators");
                Some(Integrator::PathTracing(PathTracingIntegrator {
                    min_bounces: settings.min_bounces.unwrap_or(4),
                    max_bounces,
                    world,
                    russian_roulette,
                    light_samples: 4,
                    only_direct: settings.only_direct.unwrap_or(false),
                    wavelength_bounds: bounds,
                }))
            }
        }
    }
}

#[allow(unused_variables)]
pub trait SamplerIntegrator: Sync + Send {
    fn preprocess(
        &mut self,
        sampler: &mut Box<dyn Sampler>,
        settings: &[RenderSettings],
        profile: &mut Profile,
    ) {
    }
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        camera_sample: ((f32, f32), CameraId),
        sample_id: usize,
        profile: &mut Profile,
    ) -> XYZColor;
}

pub enum Sample {
    ImageSample(XYZColor, (f32, f32)),
    LightSample(XYZColor, (f32, f32)),
}

#[allow(unused_variables)]
pub trait GenericIntegrator: Send + Sync {
    fn preprocess(
        &mut self,
        sampler: &mut Box<dyn Sampler>,
        settings: &[RenderSettings],
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
    ) -> XYZColor;
}
