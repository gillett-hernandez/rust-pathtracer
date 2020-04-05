#![allow(unused_imports, unused_variables, unused)]

pub mod config;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod math;
pub mod renderer;
pub mod world;

use config::{get_settings, RenderSettings, Settings};
use geometry::Sphere;
use integrator::{Integrator, PathTracingIntegrator};
use math::*;
use renderer::{Film, NaiveRenderer, Renderer};
use world::World;

use rayon::prelude::*;
// use std::error;
// use std::error::Error;
// use std::f32;
// use std::fmt;
// use std::env;
// use std::fs;

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

fn render(renderer: &Box<dyn Renderer>, render_settings: &RenderSettings) {
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
    let film = Film::new(width, height);
    renderer.render(&film, render_settings);
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
    let settings_vec = &config.render_settings.unwrap();
    let renderer = construct_renderer(&config, world);
    // get settings for each film
    for film in settings_vec {
        // render(integrator, &cam_setting);
        render(&renderer, &film);
    }
}
