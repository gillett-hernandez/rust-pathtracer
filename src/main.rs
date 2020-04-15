#![allow(unused_imports, unused_variables, unused)]
extern crate image;
extern crate packed_simd;

pub mod aabb;
pub mod camera;
pub mod config;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod material;
pub mod materials;
pub mod math;
pub mod renderer;
pub mod world;

use camera::{Camera, SimpleCamera};
use config::{get_settings, RenderSettings, Settings};
use geometry::{HittableList, Sphere};
use integrator::{Integrator, PathTracingIntegrator};
use material::{Material, BRDF, PDF};
use materials::{DiffuseLight, Lambertian};
use math::*;
use renderer::{Film, NaiveRenderer, Renderer};
use world::World;

use std::time::Instant;

use rand::prelude::*;
use rayon::prelude::*;

fn construct_integrator(settings: &Settings, world: World) -> Box<dyn Integrator> {
    let max_bounces = settings.max_bounces.unwrap_or(1);
    println!(
        "constructing integrator, max bounces set to {}",
        max_bounces
    );
    Box::new(PathTracingIntegrator { max_bounces, world })
}

fn construct_renderer(settings: &Settings, world: World) -> Box<dyn Renderer> {
    let integrator: Box<dyn Integrator> = construct_integrator(settings, world);
    println!("constructing renderer");
    Box::new(NaiveRenderer::new(integrator))
}

fn white_furnace_test(material: Box<dyn Material>) -> World {
    let world = World {
        bvh: Box::new(Sphere::new(5.0, Point3::new(0.0, 0.0, 0.0), Some(0))),
        background: RGBColor::new(1.0, 1.0, 1.0),
        materials: vec![material],
    };
    world
}

fn lambertian_under_lamp(color: RGBColor) -> World {
    let lambertian = Box::new(Lambertian::new(color));
    let diffuse_light = Box::new(DiffuseLight::new(RGBColor::new(1.0, 1.0, 1.0)));
    let world = World {
        bvh: Box::new(HittableList::new(vec![
            Box::new(Sphere::new(30.0, Point3::new(0.0, 0.0, -40.0), Some(1))),
            Box::new(Sphere::new(5.0, Point3::new(0.0, 0.0, 0.0), Some(0))),
        ])),
        background: RGBColor::new(0.0, 0.0, 0.0),
        materials: vec![lambertian, diffuse_light],
    };
    world
}

fn construct_scene() -> World {
    let white = RGBColor::new(1.0, 1.0, 1.0);
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
    lambertian_under_lamp(white)
}

fn render(
    renderer: &Box<dyn Renderer>,
    camera: &Box<dyn Camera>,
    render_settings: &RenderSettings,
) -> Film<RGBColor> {
    let mut film = Film::new(
        render_settings.resolution.width,
        render_settings.resolution.height,
        RGBColor::ZERO,
    );
    renderer.render(&mut film, camera, render_settings);
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
    assert!(config.output_directory != None);
    assert!(config.render_threads.unwrap() > 0);

    // let (renderer, integrator) = construct_renderer_and_integrator_from_config(config);
    // do_prerender_steps(config);

    // let cam_settings: &RenderSettings = &config.render_settings.unwrap()[0];
    // let integrator: Integrator = match (config.integrator) {
    //     Some(String::from("PT")) => PathTracingIntegrator(config),
    //     None => PathTracingIntegrator(config),
    // };
    let world = construct_scene();
    // let integrator = PathTracingIntegrator {world};
    // let settings_vec = &config.render_settings.unwrap();
    // let mut cameras = Vec::<SimpleCamera>::new();
    let mut cameras = Vec::<Box<dyn Camera>>::new();
    let camera1 = Box::new(SimpleCamera::new(
        // let camera = SimpleCamera::new(
        Point3::new(-100.0, 0.0, 0.0),
        Point3::ZERO,
        Vec3::Z,
        8.0,
        1.0,
        100.0,
        1.0,
        0.0,
        1.0,
    ));
    // );
    cameras.push(camera1);
    let camera2 = Box::new(SimpleCamera::new(
        // let camera = SimpleCamera::new(
        Point3::new(100.0, 0.0, 0.0),
        Point3::ZERO,
        Vec3::Z,
        50.0,
        1.0,
        100.0,
        1.0,
        0.0,
        1.0,
    ));
    // );
    cameras.push(camera2);
    let renderer = construct_renderer(&config, world);
    // get settings for each film
    let directory = config.output_directory.unwrap();
    for (render_id, render_settings) in config.render_settings.unwrap().iter().enumerate() {
        let camera_id = render_settings.camera_id.unwrap_or(0) as usize;

        println!(
            "starting render with film resolution {}x{}",
            render_settings.resolution.width, render_settings.resolution.height
        );

        let now = Instant::now();
        let film = render(&renderer, &cameras[camera_id], &render_settings);
        let total_pixels = film.width * film.height;
        let total_camera_rays = total_pixels * (render_settings.max_samples.unwrap() as usize);
        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
        println!("{} pixels at {} camera rays computed in {}s at {} rays per second and {} rays per second per thread", total_pixels, total_camera_rays, elapsed, (total_camera_rays as f32)/elapsed, (total_camera_rays as f32)/elapsed/(config.render_threads.unwrap() as f32));

        let now = Instant::now();
        // do stuff with film here
        let mut img: image::RgbImage =
            image::ImageBuffer::new(film.width as u32, film.height as u32);

        let mut max_luminance = 0.0;
        let mut total_luminance = 0.0;
        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.buffer[(y * film.width + x) as usize];
                let lum = Vec3::from(color).0.max_element();
                total_luminance += lum;
                if lum > max_luminance {
                    max_luminance = lum;
                }
            }
        }
        let avg_luminance = total_luminance / total_pixels as f32;
        println!(
            "computed tonemapping: max luminance {}, avg luminance {}",
            max_luminance, avg_luminance
        );
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let mut color = film.buffer[(y * film.width as u32 + x) as usize];

            //apply tonemap here
            color = color / max_luminance;

            *pixel = image::Rgb([
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
            ]);
        }
        println!("saving image...");
        img.save(format!(
            "{}/{}",
            directory,
            format!("test{}.png", render_id)
        ))
        .unwrap();
        println!("took {}s", (now.elapsed().as_millis() as f32) / 1000.0);
    }
}
