#![allow(unused_imports, unused_variables, unused)]
extern crate num_cpus;
extern crate serde;

use std::env;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
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
    pub output_directory: Option<String>,
    pub resolution: Resolution,
    pub integrator: Option<String>,
    pub selected_pair: Option<(usize, usize)>,
    pub max_bounces: Option<u16>,
    pub threads: Option<u16>,
    pub min_samples: u16,
    pub max_samples: Option<u16>,
    pub camera_id: Option<u16>,
    pub russian_roulette: Option<bool>,
    pub light_samples: Option<u16>,
    pub only_direct: Option<bool>,
}

#[derive(Deserialize, Clone)]
pub struct Settings {
    pub cameras: Vec<CameraSettings>,
    pub render_settings: Vec<RenderSettings>,
}

pub fn get_settings(filepath: String) -> Result<Settings, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(&filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let num_cpus = num_cpus::get();
    let mut settings: Settings = toml::from_str(&input)?;
    for render_settings in settings.render_settings.iter_mut() {
        render_settings.threads = match render_settings.threads {
            Some(expr) => Some(expr),
            None => Some(num_cpus as u16),
        };
    }
    return Ok(settings);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing_config() {
        let settings: Settings = match get_settings("data/config.toml".to_string()) {
            Ok(expr) => expr,
            Err(v) => {
                println!("{:?}", "couldn't read config.toml");
                println!("{:?}", v);
                return;
            }
        };
        for config in &settings.render_settings {
            assert!(config.output_directory != None);
            assert!(config.threads.unwrap() > 0)
        }
    }
}
