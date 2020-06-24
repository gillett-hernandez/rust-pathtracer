// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp)]
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
use config::{get_settings, RenderSettings, Settings};
use geometry::{AARect, Sphere};

use integrator::{Integrator, PathTracingIntegrator};
use material::Material;
use materials::*;
use math::*;
use world::*;

use renderer::{Film, NaiveRenderer, Renderer};
use tonemap::Tonemapper;

use parsing::*;

use std::sync::Arc;
use std::time::Instant;

// use rayon::prelude::*;

fn construct_integrator(settings: &RenderSettings, world: Arc<World>) -> Box<dyn Integrator> {
    let max_bounces = settings.max_bounces.unwrap_or(1);
    let russian_roulette = settings.russian_roulette.unwrap_or(true);
    let light_samples = settings.light_samples.unwrap_or(4);
    let only_direct = settings.only_direct.unwrap_or(false);
    println!(
        "constructing integrator, max bounces: {},\nrussian_roulette: {}, light_samples: {}",
        max_bounces, russian_roulette, light_samples
    );
    Box::new(PathTracingIntegrator {
        max_bounces,
        world,
        russian_roulette,
        light_samples,
        only_direct,
    })
}

fn parse_cameras_from(settings: &Settings) -> Vec<Box<dyn Camera>> {
    let mut cameras = Vec::<Box<dyn Camera>>::new();
    for camera_config in &settings.cameras {
        let camera: Box<dyn Camera> = match camera_config {
            config::CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                Box::new(SimpleCamera::new(
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

fn construct_renderer(_settings: &Settings) -> Box<dyn Renderer> {
    println!("constructing renderer");
    Box::new(NaiveRenderer::new())
}

#[allow(dead_code)]
fn white_furnace_test(material: Box<dyn Material>) -> World {
    let world = World {
        accelerator: Accelerator::new(
            vec![Sphere::new(5.0, Point3::new(0.0, 0.0, 0.0), Some(1), 0).into()],
            AcceleratorType::List,
        ),
        lights: vec![],
        background: 0,
        materials: vec![
            Box::new(DiffuseLight::new(curves::cie_e(1.0), Sidedness::Dual)),
            material,
        ],
    };
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
        load_ior_and_kappa("data/curves/copper.csv", |x: f32| x * 1000.0).unwrap();
    let (lead_ior, lead_kappa) =
        load_ior_and_kappa("data/curves/lead.csv", |x: f32| x * 1000.0).unwrap();
    let (cold_lead_ior, cold_lead_kappa) =
        load_ior_and_kappa("data/curves/lead-140K.csv", |x: f32| x * 1000.0).unwrap();
    let (platinum_ior, platinum_kappa) =
        load_ior_and_kappa("data/curves/platinum.csv", |x: f32| x * 1000.0).unwrap();

    let red = curves::red(1.0);
    let green = curves::green(1.0);
    let blue = curves::blue(1.0);
    let white = curves::cie_e(1.0);
    let moissanite = curves::cauchy(2.5, 30000.0);
    let glass = curves::cauchy(1.45, 10000.0);
    let blackbody_2000_k_illuminant = curves::blackbody(4000.0, 5.0);

    let lambertian = Box::new(Lambertian::new(color));
    let lambertian_white = Box::new(Lambertian::new(white));
    let lambertian_red = Box::new(Lambertian::new(red));
    let lambertian_green = Box::new(Lambertian::new(green));
    let lambertian_blue = Box::new(Lambertian::new(blue));
    let ggx_glass = Box::new(GGX::new(0.01, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_glass_rough = Box::new(GGX::new(0.4, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_moissanite = Box::new(GGX::new(0.01, moissanite, 1.0, flat_zero.clone(), 1.0));
    let ggx_silver_metal = Box::new(GGX::new(
        0.03,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_copper_metal = Box::new(GGX::new(
        0.03,
        copper_ior.clone(),
        1.0,
        copper_kappa.clone(),
        0.0,
    ));
    let ggx_silver_metal_rough = Box::new(GGX::new(
        0.3,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_gold_metal = Box::new(GGX::new(0.03, gold_ior, 1.0, gold_kappa, 0.0));
    let ggx_lead_metal = Box::new(GGX::new(0.03, lead_ior, 1.0, lead_kappa, 0.0));
    let ggx_cold_lead_metal = Box::new(GGX::new(0.03, cold_lead_ior, 1.0, cold_lead_kappa, 0.0));
    let ggx_platinum_metal = Box::new(GGX::new(0.03, platinum_ior, 1.0, platinum_kappa, 0.0));
    let ggx_bismuth_metal = Box::new(GGX::new(0.08, bismuth_ior, 1.0, bismuth_kappa, 0.0));

    let diffuse_light_world = Box::new(DiffuseLight::new(cie_e_world_illuminant, Sidedness::Dual));
    let diffuse_light_sphere = Box::new(DiffuseLight::new(
        blackbody_2000_k_illuminant,
        Sidedness::Reverse,
    ));

    let world = World {
        accelerator: Accelerator::new(
            vec![
                // Box::new(Sphere::new(10.0, Point3::new(0.0, 0.0, 15.0), Some(2), 0)), // big sphere light above
                (AARect::new(
                    (0.7, 0.7),
                    Point3::new(0.0, 0.0, 0.9),
                    Axis::Z,
                    false,
                    Some(2),
                    0,
                ))
                .into(),
                (AARect::new(
                    (2.0, 2.0),
                    Point3::new(0.0, 0.0, 1.0),
                    Axis::Z,
                    true,
                    Some(3),
                    1,
                ))
                .into(),
                (Sphere::new(0.3, Point3::new(-0.5, 0.0, -0.7), Some(1), 1)).into(), // ball at origin
                (Sphere::new(0.3, Point3::new(0.1, -0.5, -0.7), Some(6), 1)).into(), // ball at origin
                (Sphere::new(0.3, Point3::new(0.1, 0.5, -0.7), Some(7), 1)).into(), // ball at origin
                (AARect::new(
                    (2.0, 2.0),
                    Point3::new(0.0, 0.0, -1.0),
                    Axis::Z,
                    true,
                    Some(3),
                    2,
                ))
                .into(),
                (AARect::new(
                    (2.0, 2.0),
                    Point3::new(0.0, 1.0, 0.0),
                    Axis::Y,
                    true,
                    Some(4),
                    3,
                ))
                .into(),
                (AARect::new(
                    (2.0, 2.0),
                    Point3::new(0.0, -1.0, 0.0),
                    Axis::Y,
                    true,
                    Some(5),
                    4,
                ))
                .into(),
                (AARect::new(
                    (2.0, 2.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Axis::X,
                    true,
                    Some(3),
                    5,
                ))
                .into(),
            ],
            AcceleratorType::List,
        ),
        // the lights vector is in the form of instance indices, which means that 0 points to the first index, which in turn means it points to the lit sphere.
        lights: vec![0],
        background: 0,
        materials: vec![
            diffuse_light_world,
            ggx_moissanite,
            diffuse_light_sphere,
            lambertian_white,
            lambertian_blue,
            lambertian_red,
            ggx_gold_metal,
            ggx_copper_metal,
        ],
    };
    world
}

#[allow(unused_variables)]
fn construct_scene() -> World {
    let white = curves::cie_e(1.0);
    cornell_box(white, 0.2)
}

fn render(
    renderer: &Box<dyn Renderer>,
    camera: &Box<dyn Camera>,
    render_settings: &RenderSettings,
    world: &Arc<World>,
) -> Film<XYZColor> {
    let mut film: Film<XYZColor> = Film::new(
        render_settings.resolution.width,
        render_settings.resolution.height,
        XYZColor::BLACK,
    );
    let world_ref: Arc<World> = Arc::clone(world);
    let integrator: Arc<Box<dyn Integrator>> =
        Arc::new(construct_integrator(render_settings, world_ref));
    // let camera_ref = camera.clone();
    renderer.render(integrator.clone(), camera, render_settings, &mut film);
    film
}

fn main() -> () {
    let config: Settings = match get_settings("data/config.toml".to_string()) {
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

    let world = Arc::new(construct_scene());

    let mut cameras: Vec<Box<dyn Camera>> = parse_cameras_from(&config);
    let renderer = construct_renderer(&config);
    // get settings for each film
    for (render_id, render_settings) in config.render_settings.iter().enumerate() {
        let directory = render_settings.output_directory.as_ref();
        let camera_id = render_settings.camera_id.unwrap_or(0) as usize;

        let (width, height) = (
            render_settings.resolution.width,
            render_settings.resolution.height,
        );
        println!("starting render with film resolution {}x{}", width, height);
        let min_camera_rays = width * height * render_settings.min_samples as usize;
        println!("minimum total samples: {}", min_camera_rays);

        let aspect_ratio = width as f32 / height as f32;

        let now = Instant::now();

        &cameras[camera_id].modify_aspect_ratio(aspect_ratio);
        let film = render(&renderer, &cameras[camera_id], &render_settings, &world);

        let total_camera_rays = film.total_pixels()
            * (render_settings
                .max_samples
                .unwrap_or(render_settings.min_samples) as usize);

        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
        println!("{} pixels at {} camera rays computed in {}s at {} rays per second and {} rays per second per thread", film.total_pixels(), total_camera_rays, elapsed, (total_camera_rays as f32)/elapsed, (total_camera_rays as f32)/elapsed/(render_settings.threads.unwrap() as f32));

        let now = Instant::now();
        // do stuff with film here

        let directory_str = directory.cloned().unwrap_or(String::from("output"));
        let exr_filename = format!("{}/test{}.exr", directory_str, render_id);
        let png_filename = format!("{}/test{}.png", directory_str, render_id);

        let srgb_tonemapper = tonemap::sRGB::new(&film, 2.0);
        srgb_tonemapper.write_to_files(&film, &exr_filename, &png_filename);
        println!("took {}s", (now.elapsed().as_millis() as f32) / 1000.0);
    }
}
