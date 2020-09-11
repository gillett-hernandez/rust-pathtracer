// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp, slice_fill, vec_remove_item, partition_point)]

extern crate image;

#[macro_use]
extern crate packed_simd;

pub use math;

pub mod aabb;
pub mod accelerator;
pub mod camera;
pub mod config;
pub mod curves;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod materials;
pub mod parsing;
pub mod profile;
pub mod renderer;
pub mod texture;
pub mod tonemap;
pub mod world;

use camera::*;
use config::*;
use math::*;
use parsing::*;
use renderer::{GPUStyleRenderer, NaiveRenderer, Renderer};
use world::*;

pub const NORMAL_OFFSET: f32 = 0.00001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TransportMode {
    Radiance,
    Importance,
}

fn construct_scene(config: &Config) -> World {
    construct_world(config)
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match &*config.renderer {
        "Naive" => Box::new(NaiveRenderer::new()),
        "GPUStyle" => Box::new(GPUStyleRenderer::new()),
        _ => panic!(),
    }
}

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

    let world = construct_scene(&config);

    let cameras: Vec<Camera> = parse_cameras_from(&config);

    let renderer: Box<dyn Renderer> = construct_renderer(&config);
    renderer.render(world, cameras, &config);
}
