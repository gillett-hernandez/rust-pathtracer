#![feature(result_option_inspect, backtrace)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate packed_simd;
#[macro_use]
extern crate paste;

extern crate image;

extern crate minifb;


pub mod aabb;
pub mod accelerator;
pub mod camera;
pub mod prelude;

pub mod curves;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod materials;
pub mod mediums;
pub mod parsing;
pub mod profile;
pub mod renderer;
pub mod texture;
pub mod tonemap;
pub mod world;
