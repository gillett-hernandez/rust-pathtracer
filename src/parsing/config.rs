// use crate::prelude::*;

use std::collections::HashMap;
use std::fs::read_to_string;

use crate::parsing::tonemap::TonemapSettings;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(deny_unknown_fields)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(tag = "type")]
#[serde(deny_unknown_fields)]
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

#[allow(non_camel_case_types)]
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(deny_unknown_fields)]
pub enum ColorSpaceSettings {
    sRGB,
    Rec709,
    Rec2020,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
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
    pub camera_id: String,
    pub russian_roulette: Option<bool>,
    pub only_direct: Option<bool>,
    pub wavelength_bounds: Option<(f32, f32)>,
    pub premultiply: Option<f32>,
    pub colorspace_settings: ColorSpaceSettings,
    pub tonemap_settings: TonemapSettings,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
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
    pub premultiply: Option<f32>,
    pub colorspace_settings: ColorSpaceSettings,
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
            camera_id: data.camera_id,
            russian_roulette: data.russian_roulette,
            only_direct: data.only_direct,
            premultiply: data.premultiply,
            colorspace_settings: data.colorspace_settings,
            wavelength_bounds: data.wavelength_bounds,
            tonemap_settings: data.tonemap_settings,
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
#[serde(tag = "type")]
#[serde(deny_unknown_fields)]
pub enum RendererType {
    Naive,
    #[cfg(feature = "preview")]
    Preview {
        selected_preview_film_id: usize,
    },
    Tiled {
        tile_size: (u16, u16),
    },
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TOMLConfig {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub default_scene_file: String,
    pub renderer: RendererType,
    pub render_settings: Vec<TOMLRenderSettings>,
}

#[derive(Clone)]
pub struct Config {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub scene_file: String,
    // pub cameras: Vec<CameraSettings>,
    pub camera_names_to_index: HashMap<String, usize>,
    pub renderer: RendererType,
    pub render_settings: Vec<RenderSettings>,
}

impl Config {
    pub fn load_default() -> Self {
        let s = read_to_string("./data/config.toml").expect("failed to find default config file");
        let settings: TOMLConfig = toml::from_str(&s).expect("failed to parse default config file");
        settings.into()
    }
}

impl From<TOMLConfig> for Config {
    fn from(data: TOMLConfig) -> Self {
        Config {
            env_sampling_probability: data.env_sampling_probability,
            scene_file: data.default_scene_file,
            renderer: data.renderer,
            render_settings: data
                .render_settings
                .iter()
                .map(|e| RenderSettings::from(e.clone()))
                .collect(),
            camera_names_to_index: HashMap::new(),
        }
    }
}
