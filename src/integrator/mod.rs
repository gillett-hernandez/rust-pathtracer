mod bdpt;
pub mod gpu_style;
mod lt;
mod pt;
mod pt_hwss;
mod sppm;
pub mod utils;

pub use crate::camera::{Camera, CameraId};
use crate::config::IntegratorKind;
use crate::config::RenderSettings;
use crate::math::*;
use crate::profile::Profile;
use crate::world::World;
use math::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;
pub use pt_hwss::HWSSPathTracingIntegrator;
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

impl From<IntegratorKind> for IntegratorType {
    fn from(data: IntegratorKind) -> Self {
        match data {
            IntegratorKind::SPPM { .. } => IntegratorType::SPPM,
            IntegratorKind::PT { .. } => IntegratorType::PathTracing,
            IntegratorKind::LT { .. } => IntegratorType::LightTracing,
            IntegratorKind::BDPT { .. } => IntegratorType::BDPT,
        }
    }
}

pub enum Integrator {
    PathTracing(PathTracingIntegrator),
    HWSSPathTracing(HWSSPathTracingIntegrator),
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
        let bounds = Bounds1D::new(lower, upper);
        assert!(lower < upper);
        if settings.hwss {
            println!("constructing and returning hwss pt integrator");
            Some(Integrator::HWSSPathTracing(HWSSPathTracingIntegrator {
                inner: PathTracingIntegrator {
                    min_bounces: settings.min_bounces.unwrap_or(4),
                    max_bounces: settings.max_bounces.unwrap(),
                    world,
                    russian_roulette: settings.russian_roulette.unwrap_or(true),
                    light_samples: 4,
                    only_direct: settings.only_direct.unwrap_or(false),
                    wavelength_bounds: bounds,
                },
            }))
        } else {
            match integrator_type {
                IntegratorType::BDPT => Some(Integrator::BDPT(BDPTIntegrator {
                    max_bounces: settings.max_bounces.unwrap(),
                    world,
                    wavelength_bounds: bounds,
                })),
                IntegratorType::SPPM => Some(Integrator::SPPM(SPPMIntegrator {
                    max_bounces: settings.max_bounces.unwrap(),
                    world,
                    russian_roulette: settings.russian_roulette.unwrap_or(false),
                    camera_samples: settings.min_samples,
                    wavelength_bounds: bounds,
                    photon_map: None,
                })),
                IntegratorType::PathTracing { .. } | _ => {
                    Some(Integrator::PathTracing(PathTracingIntegrator {
                        min_bounces: settings.min_bounces.unwrap_or(4),
                        max_bounces: settings.max_bounces.unwrap(),
                        world,
                        russian_roulette: settings.russian_roulette.unwrap_or(true),
                        light_samples: 4,
                        only_direct: settings.only_direct.unwrap_or(false),
                        wavelength_bounds: bounds,
                    }))
                }
            }
        }
    }
}

pub trait SamplerIntegrator: Sync + Send {
    fn preprocess(
        &mut self,
        _sampler: &mut Box<dyn Sampler>,
        _settings: &Vec<RenderSettings>,
        _profile: &mut Profile,
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
    ) -> XYZColor;
}
