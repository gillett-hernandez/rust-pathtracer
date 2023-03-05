// use crate::prelude::*;

use crate::parsing::tonemap::TonemapSettings;

use serde::Deserialize;

use super::cameras::CameraSettings;

#[derive(Deserialize, Copy, Clone)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum IntegratorKind {
    PT {
        light_samples: u16,
        medium_aware: bool,
    },
    // BDPT {
    //     selected_pair: Option<(usize, usize)>,
    // },
    LT {
        camera_samples: u16,
    },
}

#[derive(Deserialize, Clone)]
pub struct RenderSettings {
    pub filename: Option<String>,
    pub resolution: Resolution,
    pub integrator: IntegratorKind,
    pub min_bounces: Option<u16>,
    pub max_bounces: Option<u16>,
    pub hwss: bool,
    pub threads: Option<u16>,
    pub min_samples: u16,
    pub max_samples: Option<u16>,
    pub camera_id: usize,
    pub russian_roulette: Option<bool>,
    pub only_direct: Option<bool>,
    pub wavelength_bounds: Option<(f32, f32)>,
    pub tonemap_settings: TonemapSettings,
}

#[derive(Deserialize, Clone)]
pub struct TOMLRenderSettings {
    pub filename: Option<String>,
    pub resolution: Resolution,
    pub integrator: IntegratorKind,
    pub min_bounces: Option<u16>,
    pub max_bounces: Option<u16>,
    pub hwss: bool,
    pub threads: Option<u16>,
    pub min_samples: u16,
    pub exposure: Option<f32>,
    pub max_samples: Option<u16>,
    pub camera_id: String,
    pub russian_roulette: Option<bool>,
    pub only_direct: Option<bool>,
    pub wavelength_bounds: Option<(f32, f32)>,
    pub tonemap_settings: TonemapSettings,
}

impl From<TOMLRenderSettings> for RenderSettings {
    fn from(data: TOMLRenderSettings) -> Self {
        RenderSettings {
            filename: data.filename,
            resolution: data.resolution,
            integrator: data.integrator,
            min_bounces: data.min_bounces,
            max_bounces: data.max_bounces,
            threads: data.threads,
            hwss: data.hwss,
            min_samples: data.min_samples,
            max_samples: data.max_samples,
            camera_id: 0,
            russian_roulette: data.russian_roulette,
            only_direct: data.only_direct,
            wavelength_bounds: data.wavelength_bounds,
            tonemap_settings: data.tonemap_settings,
        }
    }
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum RendererType {
    Naive,
    Preview { selected_preview_film_id: usize },
    Tiled { tile_size: (u16, u16) },
}

#[derive(Deserialize, Clone)]
pub struct TOMLConfig {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub default_scene_file: String,
    pub cameras: Vec<CameraSettings>,
    pub renderer: RendererType,
    pub render_settings: Vec<TOMLRenderSettings>,
}

#[derive(Clone)]
pub struct Config {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub scene_file: String,
    pub cameras: Vec<CameraSettings>,
    pub renderer: RendererType,
    pub render_settings: Vec<RenderSettings>,
}

impl From<TOMLConfig> for Config {
    fn from(data: TOMLConfig) -> Self {
        Config {
            env_sampling_probability: data.env_sampling_probability,
            scene_file: data.default_scene_file,
            cameras: data.cameras,
            renderer: data.renderer,
            render_settings: data
                .render_settings
                .iter()
                .map(|e| RenderSettings::from(e.clone()))
                .collect(),
        }
    }
}
