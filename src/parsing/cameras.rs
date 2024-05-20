use crate::prelude::*;

use crate::camera::{CameraEnum, PanoramaCamera, ProjectiveCamera};

#[cfg(feature = "realistic_camera")]
use crate::camera::RealisticCamera;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;


use serde::Deserialize;

use super::config::{Config, TOMLConfig};

#[cfg(feature = "realistic_camera")]
use optics::aperture::{ApertureEnum, CircularAperture, SimpleBladedAperture};
#[cfg(feature = "realistic_camera")]
use optics::parse_lenses_from;

#[cfg(feature = "realistic_camera")]
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

#[cfg(feature = "realistic_camera")]
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

#[cfg(feature = "realistic_camera")]
#[derive(Deserialize, Clone)]
pub struct RealisticCameraData {
    pub name: String,
    pub lens_spec: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,        // defaults to 0,0,1
    pub focal_adjustment: Option<f32>, // defaults to 0.0
    pub fstop: Option<f32>,            // defaults to f/2.0
    pub aperture: ApertureData,        // defaults to Circular
    // pub shutter_open_time: Option<f32>,  // defaults to 0.0
    // pub shutter_close_time: Option<f32>, // defaults to 1.0
    pub lens_zoom: Option<f32>, // defaults to 0.0
    pub radial_bins: usize,
    pub wavelength_bins: usize,
    pub sensor_size: Option<f32>, // defaults to 35mm
    pub solver_heat: Option<f32>, // defaults to 0.01
}

#[derive(Deserialize, Clone)]
pub struct SimpleCameraData {
    pub name: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>,
    pub vfov: f32,
    pub focal_distance: Option<f32>,
    pub aperture_size: Option<f32>, // in meters
    #[cfg(feature = "realistic_camera")]
    pub aperture: Option<ApertureData>, // defaults to Circular
                                    // pub shutter_open_time: Option<f32>,
                                    // pub shutter_close_time: Option<f32>,
}

#[derive(Deserialize, Clone)]
pub struct PanoramaCameraData {
    pub name: String,
    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    pub v_up: Option<[f32; 3]>, // defaults to 0,0,1
    pub fov: [f32; 2],          // in degrees. x should be in (0, 360], y should be in (0, 180]
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum CameraSettings {
    SimpleCamera(SimpleCameraData),
    PanoramaCamera(PanoramaCameraData),
    #[cfg(feature = "realistic_camera")]
    RealisticCamera(RealisticCameraData),
}

impl CameraSettings {
    pub fn get_name(&self) -> &str {
        match self {
            CameraSettings::SimpleCamera(data) => &data.name,
            CameraSettings::PanoramaCamera(data) => &data.name,
            #[cfg(feature = "realistic_camera")]
            CameraSettings::RealisticCamera(data) => &data.name,
        }
    }
}

pub fn parse_config_and_cameras(settings: TOMLConfig) -> (Config, Vec<CameraEnum>) {
    // this function is necessary because different render settings change the aspect ratio of the camera,
    // even if they're using the "same" camera
    let mut cameras: Vec<CameraEnum> = Vec::new();
    let mut camera_map: HashMap<String, CameraEnum> = HashMap::new();
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
        let (name, camera): (String, CameraEnum) = match camera_config {
            CameraSettings::SimpleCamera(cam) => {
                // let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);

                (
                    cam.name.clone(),
                    CameraEnum::ProjectiveCamera(ProjectiveCamera::new(
                        Point3::from(cam.look_from),
                        Point3::from(cam.look_at),
                        Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])).normalized(),
                        cam.vfov,
                        cam.focal_distance.unwrap_or(10.0),
                        cam.aperture_size.unwrap_or(0.0),
                        // shutter_open_time,
                        // cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
                    )),
                )
            }
            CameraSettings::PanoramaCamera(cam) => (
                cam.name.clone(),
                CameraEnum::PanoramaCamera(PanoramaCamera::new(
                    Point3::from(cam.look_from),
                    Point3::from(cam.look_at),
                    Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])).normalized(),
                    cam.fov[0],
                    cam.fov[1],
                )),
            ),
            #[cfg(feature = "realistic_camera")]
            CameraSettings::RealisticCamera(cam) => {
                let mut camera_file = File::open(&cam.lens_spec).unwrap();
                let mut camera_spec = String::new();
                camera_file.read_to_string(&mut camera_spec).unwrap();
                let (interfaces, _n0, _n1) = parse_lenses_from(&camera_spec);

                // let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                (
                    cam.name.clone(),
                    CameraEnum::RealisticCamera(RealisticCamera::new(
                        Point3::from(cam.look_from),
                        Point3::from(cam.look_at),
                        Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])).normalized(),
                        cam.focal_adjustment.unwrap_or(0.0),
                        cam.sensor_size.unwrap_or(35.0),
                        cam.fstop.unwrap_or(2.0),
                        cam.lens_zoom.unwrap_or(0.0),
                        interfaces,
                        cam.aperture.into(),
                        // shutter_open_time,
                        // cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
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
