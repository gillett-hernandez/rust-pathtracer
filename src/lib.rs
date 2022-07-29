#![feature(result_option_inspect)]
#[macro_use]
extern crate log;

extern crate image;

#[macro_use]
extern crate packed_simd;

extern crate minifb;

pub use math;

pub mod aabb;
pub mod accelerator;
pub mod camera;

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

pub fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}
