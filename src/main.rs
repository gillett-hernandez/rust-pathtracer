// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp, slice_fill)]
extern crate image;

#[macro_use]
extern crate packed_simd;

pub mod aabb;
pub mod accelerator;
pub mod camera;
pub mod config;
pub mod curves;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod material;
pub mod materials;
pub mod math;

pub mod parsing;
pub mod renderer;
pub mod tonemap;
pub mod world;

use camera::{Camera, SimpleCamera};
use config::{get_settings, Config};
use geometry::{AARect, Aggregate, Instance, Sphere};

// use materials::*;
use math::*;
use world::*;

use renderer::{NaiveRenderer, Renderer};

use parsing::*;

// use integrator::*;
// use std::sync::Arc;
// use std::time::Instant;

// use rayon::prelude::*;

fn parse_cameras_from(settings: &Config) -> Vec<Camera> {
    let mut cameras = Vec::<Camera>::new();
    for camera_config in &settings.cameras {
        let camera: Camera = match camera_config {
            config::CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                Camera::SimpleCamera(SimpleCamera::new(
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

#[allow(dead_code)]
fn white_furnace_test(material: MaterialEnum) -> World {
    let world = World::new(
        vec![Instance::from(Aggregate::from(Sphere::new(
            5.0,
            Point3::new(0.0, 0.0, 0.0),
            MaterialId::Material(0),
            0,
        )))],
        vec![material],
        EnvironmentMap::new(curves::cie_e(1.0)),
    );
    world
}

#[allow(unused_variables)]
fn cornell_box(color: SPD, world_strength: f32) -> World {
    let cie_e_world_illuminant = curves::cie_e(world_strength);
    let flat_zero = curves::void();
    let flat_one = curves::cie_e(1.0);
    let cie_e_illuminant = curves::cie_e(15.0);
    let (silver_ior, silver_kappa) =
        load_ior_and_kappa("data/curves/silver.csv", |x: f32| x * 1000.0).unwrap();
    let (gold_ior, gold_kappa) =
        load_ior_and_kappa("data/curves/gold.csv", |x: f32| x * 1000.0).unwrap();

    let (bismuth_ior, bismuth_kappa) =
        load_ior_and_kappa("data/curves/bismuth.csv", |x: f32| x * 1000.0).unwrap();
    let (copper_ior, copper_kappa) =
        load_ior_and_kappa("data/curves/copper-mcpeak.csv", |x: f32| x * 1000.0).unwrap();
    let (lead_ior, lead_kappa) =
        load_ior_and_kappa("data/curves/lead.csv", |x: f32| x * 1000.0).unwrap();
    let (cold_lead_ior, cold_lead_kappa) =
        load_ior_and_kappa("data/curves/lead-140K.csv", |x: f32| x * 1000.0).unwrap();
    let (platinum_ior, platinum_kappa) =
        load_ior_and_kappa("data/curves/platinum.csv", |x: f32| x * 1000.0).unwrap();
    let (iron_ior, iron_kappa) =
        load_ior_and_kappa("data/curves/iron-johnson.csv", |x: f32| x * 1000.0).unwrap();

    let red = curves::red(1.0);
    let green = curves::green(1.0);
    let blue = curves::blue(1.0);
    let white = curves::cie_e(1.0);
    let moissanite = curves::cauchy(2.5, 40000.0);
    let glass = curves::cauchy(1.45, 10000.0);
    let blackbody_illuminant = curves::blackbody(5500.0, 5.0);

    let lambertian = MaterialEnum::from(Lambertian::new(color));
    let lambertian_white = MaterialEnum::from(Lambertian::new(white));
    let lambertian_red = MaterialEnum::from(Lambertian::new(red));
    let lambertian_green = MaterialEnum::from(Lambertian::new(green));
    let lambertian_blue = MaterialEnum::from(Lambertian::new(blue));
    let ggx_glass = MaterialEnum::from(GGX::new(0.01, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_glass_rough =
        MaterialEnum::from(GGX::new(0.4, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_moissanite =
        MaterialEnum::from(GGX::new(0.01, moissanite, 1.0, flat_zero.clone(), 1.0));
    let ggx_silver_metal = MaterialEnum::from(GGX::new(
        0.03,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_copper_metal = MaterialEnum::from(GGX::new(
        0.03,
        copper_ior.clone(),
        1.0,
        copper_kappa.clone(),
        0.0,
    ));
    let ggx_silver_metal_rough = MaterialEnum::from(GGX::new(
        0.08,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_gold_metal = MaterialEnum::from(GGX::new(0.03, gold_ior, 1.0, gold_kappa, 0.0));
    let ggx_lead_metal = MaterialEnum::from(GGX::new(0.03, lead_ior, 1.0, lead_kappa, 0.0));
    let ggx_cold_lead_metal =
        MaterialEnum::from(GGX::new(0.03, cold_lead_ior, 1.0, cold_lead_kappa, 0.0));
    let ggx_platinum_metal =
        MaterialEnum::from(GGX::new(0.03, platinum_ior, 1.0, platinum_kappa, 0.0));
    let ggx_bismuth_metal =
        MaterialEnum::from(GGX::new(0.08, bismuth_ior, 1.0, bismuth_kappa, 0.0));
    let ggx_iron_metal = MaterialEnum::from(GGX::new(0.08, iron_ior, 1.0, iron_kappa, 0.0));

    let env_map = EnvironmentMap::new(cie_e_world_illuminant);
    let diffuse_light =
        MaterialEnum::from(DiffuseLight::new(blackbody_illuminant, Sidedness::Dual));

    let world = World::new(
        vec![
            Instance::from(Aggregate::from(AARect::new(
                (0.7, 0.7),
                Point3::new(0.0, 0.0, 0.9),
                Axis::Z,
                false,
                MaterialId::Light(1),
                0,
            ))),
            Instance::from(Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 0.0, 1.0),
                Axis::Z,
                true,
                2.into(),
                1,
            ))),
            Instance::from(Aggregate::from(Sphere::new(
                0.3,
                Point3::new(-0.5, 0.0, -0.7),
                0.into(),
                2,
            ))), // ball at origin
            Instance::from(Aggregate::from(Sphere::new(
                0.3,
                Point3::new(0.1, -0.5, -0.7),
                5.into(),
                3,
            ))), // ball at origin
            Instance::from(Aggregate::from(Sphere::new(
                0.3,
                Point3::new(0.1, 0.5, -0.7),
                6.into(),
                4,
            ))), // ball at origin
            Instance::from(Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 0.0, -1.0),
                Axis::Z,
                true,
                2.into(),
                5,
            ))),
            Instance::from(Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 1.0, 0.0),
                Axis::Y,
                true,
                3.into(),
                6,
            ))),
            Instance::from(Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, -1.0, 0.0),
                Axis::Y,
                true,
                4.into(),
                7,
            ))),
            Instance::from(Aggregate::from(AARect::new(
                // back wall
                (2.0, 2.0),
                Point3::new(1.0, 0.0, 0.0),
                Axis::X,
                true,
                2.into(),
                8,
            ))),
        ],
        vec![
            ggx_glass,
            diffuse_light,
            lambertian_white,
            lambertian_blue,
            lambertian_red,
            ggx_gold_metal,
            ggx_copper_metal,
        ],
        env_map,
    );
    world
}

#[allow(unused_variables)]
fn construct_scene() -> World {
    let white = curves::cie_e(1.0);
    cornell_box(white, 0.2)
}

// fn render(
//     renderer: &Box<dyn Renderer>,
//     camera: &Box<dyn Camera>,
//     render_settings: &RenderSettings,
//     world: &Arc<World>,
// ) -> Film<XYZColor> {

// }

fn main() -> () {
    let config: Config = match get_settings("data/config.toml".to_string()) {
        Ok(expr) => expr,
        Err(v) => {
            println!("{:?}", "couldn't read config.toml");
            println!("{:?}", v);
            return;
        }
    };
    let threads = config
        .render_settings
        .iter()
        .map(|i| &i.threads)
        .fold(1, |a, &b| a.max(b.unwrap_or(1)));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();

    // do_prerender_steps(config);

    let world = construct_scene();

    let cameras: Vec<Camera> = parse_cameras_from(&config);
    // some integrators only work with certain renderers.
    // collect the render settings bundles that apply to certain integrators, and correlate them with their corresponding renderers.
    // use multiple renderers if necessary

    // let renderer = construct_renderer(&config);
    let renderer = NaiveRenderer::new();
    renderer.render(world, cameras, &config);
}
