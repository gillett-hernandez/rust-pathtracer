extern crate num_cpus;
extern crate serde;

// use std::env;
use std::fs::File;
use std::io::Read;
// use std::io::{self, BufWriter, Write};
// use std::path::Path;

use serde::Deserialize;
use toml;

#[derive(Deserialize, Copy, Clone)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Deserialize, Copy, Clone)]
pub struct SimpleCameraSettings {
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,
    pub vfov: f32,
    pub focal_distance: Option<f32>,
    pub aperture_size: Option<f32>,
    pub shutter_open_time: Option<f32>,
    pub shutter_close_time: Option<f32>,
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum CameraSettings {
    SimpleCamera(SimpleCameraSettings),
}
#[derive(Deserialize, Clone)]
pub struct RenderSettings {
    pub filename: Option<String>,
    pub resolution: Resolution,
    pub integrator: Option<String>,
    pub selected_pair: Option<(usize, usize)>,
    pub max_bounces: Option<u16>,
    pub threads: Option<u16>,
    pub min_samples: u16,
    pub exposure: Option<f32>,
    pub max_samples: Option<u16>,
    pub camera_id: Option<u16>,
    pub russian_roulette: Option<bool>,
    pub light_samples: Option<u16>,
    pub only_direct: Option<bool>,
    pub wavelength_bounds: Option<(f32, f32)>,
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub env_strength: Option<f32>,             //defaults to 0.5
    pub cameras: Vec<CameraSettings>,
    pub render_settings: Vec<RenderSettings>,
}
