#![allow(unused_imports, unused_variables, unused)]
#![feature(clamp)]
extern crate image;

#[macro_use]
extern crate packed_simd;

pub mod aabb;
pub mod camera;
pub mod config;
pub mod geometry;
pub mod hittable;
pub mod illuminants_and_colors;
pub mod integrator;
pub mod material;
pub mod materials;
pub mod math;
pub mod renderer;
pub mod tonemap;
pub mod world;

use camera::{Camera, SimpleCamera};
use config::{get_settings, RenderSettings, Settings};
use geometry::{HittableList, Sphere, AARect};

use integrator::{Integrator, PathTracingIntegrator};
use material::{Material, BRDF, PDF};
use materials::{DiffuseLight, Lambertian};
use math::*;
use renderer::{Film, NaiveRenderer, Renderer};
use world::World;

use std::sync::Arc;
use std::time::Instant;

use rand::prelude::*;
use rayon::prelude::*;

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

fn construct_renderer(settings: &Settings) -> Box<dyn Renderer> {
    println!("constructing renderer");
    Box::new(NaiveRenderer::new())
}

fn white_furnace_test(material: Box<dyn Material>) -> World {
    let world = World {
        bvh: Box::new(HittableList::new(vec![Box::new(Sphere::new(
            5.0,
            Point3::new(0.0, 0.0, 0.0),
            Some(1),
            0,
        ))])),
        lights: vec![],
        background: 0,
        materials: vec![
            Box::new(DiffuseLight::new(illuminants_and_colors::cie_e(1.0))),
            material,
        ],
    };
    world
}

fn lambertian_under_lamp(color: SPD, world_strength: f32) -> World {
    let cie_e_world_illuminant = illuminants_and_colors::cie_e(world_strength);
    let cie_e_illuminant = illuminants_and_colors::cie_e(15.0);

    let red = illuminants_and_colors::red(1.0);
    let green = illuminants_and_colors::green(1.0);
    let blue = illuminants_and_colors::blue(1.0);
    let white = illuminants_and_colors::cie_e(1.0);
    let blackbody_2000_k_illuminant = illuminants_and_colors::blackbody(3500.0, 10.0);

    let lambertian = Box::new(Lambertian::new(color));
    let lambertian_white = Box::new(Lambertian::new(white));
    let lambertian_red = Box::new(Lambertian::new(red));
    let lambertian_green = Box::new(Lambertian::new(green));
    let lambertian_blue = Box::new(Lambertian::new(blue));

    let diffuse_light_world = Box::new(DiffuseLight::new(cie_e_world_illuminant));
    let diffuse_light_sphere = Box::new(DiffuseLight::new(blackbody_2000_k_illuminant));


    let world = World {
        bvh: Box::new(HittableList::new(vec![
            // Box::new(Sphere::new(10.0, Point3::new(0.0, 0.0, 15.0), Some(2), 0)), // big sphere light above
            Box::new(AARect::new((0.5, 0.5), Point3::new(0.0, 0.0, 0.9), Axis::Z, Some(2), 0)),
            Box::new(AARect::new((2.0, 2.0), Point3::new(0.0, 0.0, 1.0), Axis::Z, Some(3), 1)),
            Box::new(Sphere::new(0.3, Point3::new(0.0, 0.0, -0.3), Some(1), 1)), // ball at origin
            Box::new(AARect::new((2.0, 2.0), Point3::new(0.0, 0.0, -1.0), Axis::Z, Some(3), 2)),
            Box::new(AARect::new((2.0, 2.0), Point3::new(0.0, 1.0, 0.0), Axis::Y, Some(4), 3)),
            Box::new(AARect::new((2.0, 2.0), Point3::new(0.0, -1.0, 0.0), Axis::Y, Some(5), 4)),
            Box::new(AARect::new((2.0, 2.0), Point3::new(1.0, 0.0, 0.0), Axis::X, Some(3), 5)),
        ])),
        // the lights vector is in the form of instance indices, which means that 0 points to the first index, which in turn means it points to the lit sphere.
        lights: vec![0],
        background: 0,
        materials: vec![diffuse_light_world, lambertian, diffuse_light_sphere, lambertian_white, lambertian_blue, lambertian_red],
    };
    world
}

fn construct_scene() -> World {
    let white = illuminants_and_colors::cie_e(1.0);
    let red = illuminants_and_colors::red(1.0);
    let green = illuminants_and_colors::green(1.0);
    let blue = illuminants_and_colors::blue(1.0);
    // let lambertian = Box::new(Lambertian::new(white));
    // let diffuse_light = Box::new(DiffuseLight::new());
    // let world = World {
    //     bvh: Box::new(HittableList::new(vec![
    //         Box::new(Sphere::new(30.0, Point3::new(0.0, 0.0, 40.0), Some(0))),
    //         // Box::new(Sphere::new(30.0, Point3::new(0.0, 0.0, -40.0), Some(1))),
    //         Box::new(Sphere::new(30.0, Point3::new(0.0, 0.0, -40.0), Some(0))),
    //         Box::new(Sphere::new(5.0, Point3::new(0.0, 0.0, 0.0), Some(0))),
    //     ])),
    //     // background: RGBColor::new(0.0, 0.0, 0.0),
    //     background: white,
    //     materials: vec![lambertian, diffuse_light],
    // };
    // world
    // let lambertian = Box::new(Lambertian::new(white));
    // white_furnace_test(lambertian)
    lambertian_under_lamp(white, 0.0)
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
    let camera_ref = camera.clone();
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

        let aspect_ratio = width as f32 / height as f32;

        let now = Instant::now();

        &cameras[camera_id].modify_aspect_ratio(aspect_ratio);
        let film = render(&renderer, &cameras[camera_id], &render_settings, &world);

        let total_pixels = film.width * film.height;
        let total_camera_rays = total_pixels
            * (render_settings
                .max_samples
                .unwrap_or(render_settings.min_samples) as usize);

        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;

        let now = Instant::now();
        // do stuff with film here
        let mut img: image::RgbImage =
        image::ImageBuffer::new(film.width as u32, film.height as u32);

        let mut max_luminance = 0.0;
        let mut total_luminance = 0.0;
        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.buffer[(y * film.width + x) as usize];
                let lum = color.y();
                assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                total_luminance += lum;
                if lum > max_luminance {
                    println!("max lum so far was {} and occurred at ({}, {})", max_luminance, x, y);
                    max_luminance = lum;
                }
            }
        }
        println!("{} pixels at {} camera rays computed in {}s at {} rays per second and {} rays per second per thread", total_pixels, total_camera_rays, elapsed, (total_camera_rays as f32)/elapsed, (total_camera_rays as f32)/elapsed/(render_settings.threads.unwrap() as f32));
        let avg_luminance = total_luminance / total_pixels as f32;
        // let exposure = 1.0f32;
        // max_luminance = exposure^(-1/gamma)
        // max_luminance = 1 / exposure^(1/gamma)
        let gamma = 2.2f32.recip();
        let exposure = 2.0 * avg_luminance.recip().powf(gamma);

        // let upper = exposure.powf(-1.0/gamma);
        println!(
            "computed tonemapping: max luminance {}, avg luminance {}, exposure is {}",
            max_luminance, avg_luminance, exposure
        );

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let mut color = film.buffer[(y * film.width as u32 + x) as usize];

            //apply tonemap here
            let pixel_luminance = color.y();
            let target_luminance = exposure * pixel_luminance.powf(gamma);
            color.0 = color.0 * target_luminance / pixel_luminance;
            let [r, g, b, _]: [f32; 4] = RGBColor::from(color).0.into();

            *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
        }
        println!("saving image...");
        img.save(format!(
            "{}/{}",
            directory.unwrap().clone(),
            format!("test{}.png", render_id)
        ))
        .unwrap();
        println!("took {}s", (now.elapsed().as_millis() as f32) / 1000.0);
    }
}
