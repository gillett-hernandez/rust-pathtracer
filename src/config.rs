extern crate num_cpus;
extern crate serde;

use crate::camera::{Camera, ProjectiveCamera};
use crate::math::{Point3, Vec3};

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use serde::Deserialize;
use toml;

#[derive(Deserialize, Copy, Clone)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Deserialize, Clone)]
pub struct SimpleCameraSettings {
    pub name: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,
    pub vfov: f32,
    pub focal_distance: Option<f32>,
    pub aperture_size: Option<f32>,
    pub shutter_open_time: Option<f32>,
    pub shutter_close_time: Option<f32>,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum CameraSettings {
    SimpleCamera(SimpleCameraSettings),
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum IntegratorKind {
    PT {
        light_samples: Option<u16>,
    },
    BDPT {
        selected_pair: Option<(usize, usize)>,
    },
    LT {
        camera_samples: Option<u16>,
    },
    SPPM {
        photon_cache_size: Option<usize>,
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
    pub exposure: Option<f32>,
    pub max_samples: Option<u16>,
    pub camera_id: usize,
    pub russian_roulette: Option<bool>,
    pub only_direct: Option<bool>,
    pub wavelength_bounds: Option<(f32, f32)>,
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
            exposure: data.exposure,
            max_samples: data.max_samples,
            camera_id: 0,
            russian_roulette: data.russian_roulette,
            only_direct: data.only_direct,
            wavelength_bounds: data.wavelength_bounds,
        }
    }
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum RendererType {
    Naive,
    GPUStyle {
        tile_width: usize,
        tile_height: usize,
    },
    /* Tiled {
        tile_width: usize,
        tile_height: usize,
    }, */
    Preview {
        selected_preview_film_id: usize,
    },
}

#[derive(Deserialize, Clone)]
pub struct TOMLConfig {
    pub env_sampling_probability: Option<f32>, //defaults to 0.5
    pub scene_file: String,
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
            scene_file: data.scene_file,
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

pub fn parse_cameras_from(settings: &TOMLConfig) -> (Config, Vec<Camera>) {
    let mut cameras: Vec<Camera> = Vec::new();
    let mut camera_map: HashMap<String, Camera> = HashMap::new();
    let mut config = Config::from(settings.clone());
    for camera_config in &settings.cameras {
        let (name, camera): (String, Camera) = match camera_config {
            CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                (
                    cam.name.clone(),
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
                    )),
                )
            }
        };
        camera_map.insert(name, camera);
    }
    for (render_settings, toml_settings) in config
        .render_settings
        .iter_mut()
        .zip(settings.render_settings.iter())
    {
        let cam_id = cameras.len();
        let camera = camera_map[&toml_settings.camera_id].clone();
        render_settings.camera_id = cam_id;
        cameras.push(camera);
    }
    (config, cameras)
}

pub fn get_settings(filepath: String) -> Result<TOMLConfig, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(&filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let num_cpus = num_cpus::get();
    let mut settings: TOMLConfig = toml::from_str(&input)?;
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
        let settings: TOMLConfig = match get_settings("data/config.toml".to_string()) {
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
