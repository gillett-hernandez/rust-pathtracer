extern crate num_cpus;
extern crate serde;

use crate::camera::{Camera, ProjectiveCamera};
use crate::math::{Point3, Vec3};

use std::fs::File;
use std::io::Read;

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
    pub min_bounces: Option<u16>,
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
    pub tile_width: usize,
    pub tile_height: usize,
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub scene_file: String,
    pub cameras: Vec<CameraSettings>,
    pub renderer: String,
    pub render_settings: Vec<RenderSettings>,
}

pub fn parse_cameras_from(settings: &Config) -> Vec<Camera> {
    let mut cameras = Vec::<Camera>::new();
    for camera_config in &settings.cameras {
        let camera: Camera = match camera_config {
            CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                Camera::ProjectiveCamera(ProjectiveCamera::new(
                    Point3::from(cam.look_from),
                    Point3::from(cam.look_at),
                    Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])),
                    cam.vfov,
                    1.0,
                    cam.focal_distance.unwrap_or(10.0),
                    cam.aperture_size.unwrap_or(0.0),
                    shutter_open_time,
                    cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
                ))
            }
        };
        cameras.push(camera);
    }
    cameras
}

pub fn get_settings(filepath: String) -> Result<Config, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(&filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let num_cpus = num_cpus::get();
    let mut settings: Config = toml::from_str(&input)?;
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
        let settings: Config = match get_settings("data/config.toml".to_string()) {
            Ok(expr) => expr,
            Err(v) => {
                println!("{:?}", "couldn't read config.toml");
                println!("{:?}", v);
                return;
            }
        };
        for config in &settings.render_settings {
            assert!(config.filename != None);
            assert!(config.threads.unwrap() > 0)
        }
    }
}
