// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp, slice_fill, vec_remove_item)]

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
pub mod texture;
pub mod tonemap;
pub mod world;

use camera::*;
use config::*;
use math::*;
use parsing::*;
use renderer::{NaiveRenderer, Renderer};
use world::*;

pub const NORMAL_OFFSET: f32 = 0.00001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

#[derive(Copy, Clone, PartialEq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

fn construct_scene(config: &Config) -> World {
    construct_world(config)
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

    let renderer = NaiveRenderer::new();
    renderer.render(world, cameras, &config);
}
