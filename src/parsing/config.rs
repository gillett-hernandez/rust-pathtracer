use crate::camera::{Camera, ProjectiveCamera, RealisticCamera};
use crate::math::{Point3, Vec3};
use crate::parsing::tonemap::TonemapSettings;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use optics::aperture::{ApertureEnum, CircularAperture, SimpleBladedAperture};
use optics::parse_lenses_from;
use serde::Deserialize;
use toml;

#[derive(Deserialize, Copy, Clone)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(tag = "type")]
pub enum ApertureData {
    Circular,
    // only 6 blades are supported right now.
    // sharpness should be between -2 and 2.3
    // 2.3 is very close to a circular aperture
    // -2 is something like a pointy star
    Bladed { blades: u8, sharpness: f32 },
}

impl Into<ApertureEnum> for ApertureData {
    fn into(self) -> ApertureEnum {
        match self {
            ApertureData::Circular => ApertureEnum::CircularAperture(CircularAperture::default()),
            ApertureData::Bladed { blades, sharpness } => {
                ApertureEnum::SimpleBladedAperture(SimpleBladedAperture::new(blades, sharpness))
            }
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct SimpleCameraSettings {
    pub name: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,
    pub vfov: f32,
    pub focal_distance: Option<f32>,
    pub aperture_size: Option<f32>,     // in meters
    pub aperture: Option<ApertureData>, // defaults to Circular
    pub shutter_open_time: Option<f32>,
    pub shutter_close_time: Option<f32>,
}

#[derive(Deserialize, Clone)]
pub struct RealisticCameraSettings {
    pub name: String,
    pub lens_spec: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,          // defaults to 0,0,1
    pub focal_adjustment: Option<f32>,   // defaults to 0.0
    pub fstop: Option<f32>,              // defaults to f/2.0
    pub aperture: ApertureData,          // defaults to Circular
    pub shutter_open_time: Option<f32>,  // defaults to 0.0
    pub shutter_close_time: Option<f32>, // defaults to 1.0
    pub lens_zoom: Option<f32>,          // defaults to 0.0
    pub radial_bins: usize,
    pub wavelength_bins: usize,
    pub sensor_size: Option<f32>, // defaults to 35mm
    pub solver_heat: Option<f32>, // defaults to 0.01
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum CameraSettings {
    SimpleCamera(SimpleCameraSettings),
    RealisticCamera(RealisticCameraSettings),
}

impl CameraSettings {
    pub fn get_name(&self) -> &str {
        match self {
            CameraSettings::SimpleCamera(data) => &data.name,
            CameraSettings::RealisticCamera(data) => &data.name,
        }
    }
}

#[derive(Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum IntegratorKind {
    PT {
        light_samples: u16,
    },
    BDPT {
        selected_pair: Option<(usize, usize)>,
    },
    LT {
        camera_samples: u16,
    },
    // SPPM {
    //     photon_cache_size: usize,
    // },
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
    // SPPM,
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

pub fn parse_cameras_from(settings: TOMLConfig) -> (Config, Vec<Camera>) {
    // this function is necessary because different render settings change the aspect ratio of the camera,
    // even if they're using the "same" camera
    let mut cameras: Vec<Camera> = Vec::new();
    let mut camera_map: HashMap<String, Camera> = HashMap::new();
    let mut config = Config::from(settings.clone());
    let mut camera_ids = Vec::new();

    for toml_settings in settings.render_settings.iter() {
        if !camera_ids.contains(&toml_settings.camera_id) {
            camera_ids.push(toml_settings.camera_id.clone());
        }
    }
    for camera_config in &settings.cameras {
        if !camera_ids.contains(&camera_config.get_name().to_owned()) {
            continue;
        }
        let (name, camera): (String, Camera) = match camera_config {
            CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);

                (
                    cam.name.clone(),
                    Camera::ProjectiveCamera(ProjectiveCamera::new(
                        Point3::from(cam.look_from),
                        Point3::from(cam.look_at),
                        Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])).normalized(),
                        cam.vfov,
                        cam.focal_distance.unwrap_or(10.0),
                        cam.aperture_size.unwrap_or(0.0),
                        shutter_open_time,
                        cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
                    )),
                )
            }
            CameraSettings::RealisticCamera(cam) => {
                let mut camera_file = File::open(&cam.lens_spec).unwrap();
                let mut camera_spec = String::new();
                camera_file.read_to_string(&mut camera_spec).unwrap();
                let (interfaces, _n0, _n1) = parse_lenses_from(&camera_spec);

                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                (
                    cam.name.clone(),
                    Camera::RealisticCamera(RealisticCamera::new(
                        Point3::from(cam.look_from),
                        Point3::from(cam.look_at),
                        Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])).normalized(),
                        cam.focal_adjustment.unwrap_or(0.0),
                        cam.sensor_size.unwrap_or(35.0),
                        cam.fstop.unwrap_or(2.0),
                        cam.lens_zoom.unwrap_or(0.0),
                        interfaces,
                        cam.aperture.into(),
                        shutter_open_time,
                        cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
                        cam.radial_bins,
                        cam.wavelength_bins,
                        cam.solver_heat.unwrap_or(0.01),
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
        let camera = camera_map[&toml_settings.camera_id]
            .clone()
            .with_aspect_ratio(
                render_settings.resolution.width as f32 / render_settings.resolution.height as f32,
            );
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
    let num_cpus = num_cpus::get();
    let mut settings: TOMLConfig = toml::from_str(&input)?;
    for render_settings in settings.render_settings.iter_mut() {
        render_settings.threads = match render_settings.threads {
            Some(expr) => Some(expr),
            None => Some(num_cpus as u16),
        };
    }
    Ok(settings)
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
