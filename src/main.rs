#![allow(unused_imports, unused_variables, unused)]
extern crate image;

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
use geometry::Sphere;
use integrator::{Integrator, PathTracingIntegrator};
use materials::Lambertian;
use math::*;
use renderer::{Film, NaiveRenderer, Renderer};
use world::World;

// use image::{GenericImageView, ImageBuffer};

use rand::prelude::*;
use rayon::prelude::*;

// use std::collections::HashMap;
// use std::fmt;
// use std::fs::File;
// use std::io::prelude::*;
// use std::io::BufReader;
// use std::sync::Arc;

fn construct_integrator(settings: &Settings, world: World) -> Box<dyn Integrator> {
    Box::new(PathTracingIntegrator {
        max_bounces: settings.max_bounces.unwrap_or(1),
        world,
    })
}

fn construct_renderer(settings: &Settings, world: World) -> Box<dyn Renderer> {
    let integrator: Box<dyn Integrator> = construct_integrator(settings, world);
    Box::new(NaiveRenderer::new(integrator))
}

fn render(
    renderer: &Box<dyn Renderer>,
    camera: &Box<dyn Camera>,
    render_settings: &RenderSettings,
) -> Film<RGBColor> {
    let width = match render_settings.resolution {
        Some(res) => res.width,
        None => 512,
    };
    let height = match render_settings.resolution {
        Some(res) => res.height,
        None => 512,
    };
    println!(
        "starting render with film resolution {:?}x{:?}",
        width, height
    );
    let mut film = Film::new(width, height, RGBColor::ZERO);
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
    let world = World {
        bvh: Box::new(Sphere {
            radius: 1.0,
            origin: Point3::ZERO,
        }),
        background: RGBColor::new(0.2, 0.3, 0.2),
    };
    // let integrator = PathTracingIntegrator {world};
    // let settings_vec = &config.render_settings.unwrap();
    // let mut cameras = Vec::<SimpleCamera>::new();
    let mut cameras = Vec::<Box<dyn Camera>>::new();
    let camera = Box::new(SimpleCamera::new(
        // let camera = SimpleCamera::new(
        Point3::new(-100.0, 0.0, 0.0),
        Point3::ZERO,
        Vec3::Z,
        5.0,
        1.0,
        100.0,
        1.0,
        0.0,
        1.0,
    ));
    // );
    cameras.push(camera);
    let renderer = construct_renderer(&config, world);
    // get settings for each film
    // (cameras[0]).get_ray(0.0, 0.0);
    let directory = config.output_directory.unwrap();
    for render_settings in config.render_settings.unwrap() {
        let film = render(
            &renderer,
            &cameras[render_settings.camera_id.unwrap_or(0) as usize],
            &render_settings,
        );

        // do stuff with film here
        let mut img: image::RgbImage =
            image::ImageBuffer::new(film.width as u32, film.height as u32);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let mut color = film.buffer[(y * film.width as u32 + x) as usize];

            *pixel = image::Rgb([
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
            ]);
        }
        img.save(format!("{}/{}", directory, "test.png")).unwrap();
    }
}
