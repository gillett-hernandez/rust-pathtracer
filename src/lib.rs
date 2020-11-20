// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp, slice_fill, vec_remove_item, partition_point)]

extern crate image;

#[macro_use]
extern crate packed_simd;

extern crate minifb;

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
