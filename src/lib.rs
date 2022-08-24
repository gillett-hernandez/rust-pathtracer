#![feature(result_option_inspect)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate packed_simd;
#[macro_use]
extern crate paste;
extern crate image;
extern crate minifb;

use minifb::{Key, Window, WindowOptions};
use rayon::prelude::*;
use renderer::Film;
use tonemap::{Converter, Tonemapper};

use math::{
    prelude::{XYZColor, PDF},
    traits::{Area, Field, Measure, ProjectedSolidAngle, Throughput, ToScalar},
};

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

// mauve. universal sign of danger
pub const MAUVE: XYZColor = XYZColor::new(0.5199467, 51.48687, 1.0180528);

pub const NORMAL_OFFSET: f32 = 0.001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TransportMode {
    Radiance,
    Importance,
}

impl Default for TransportMode {
    fn default() -> Self {
        TransportMode::Importance
    }
}

pub fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

pub fn window_loop<F>(
    width: usize,
    height: usize,
    max_framerate: usize,
    options: WindowOptions,
    mut func: F,
) where
    F: FnMut(&Window, &mut Vec<u32>, usize, usize) -> (),
{
    let mut window = Window::new("Window", width, height, options).unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(
        (1000000 / max_framerate) as u64,
    )));

    let mut film = Film::new(width, height, 0u32);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        film.buffer.fill(0u32);
        func(&window, &mut film.buffer, width, height);

        window
            .update_with_buffer(&film.buffer, width, height)
            .unwrap();
    }
}

pub fn update_window_buffer(
    buffer: &mut [u32],
    film: &Film<XYZColor>,
    tonemapper: &mut dyn Tonemapper,
    converter: Converter,
    factor: f32,
) {
    let width = film.width;
    debug_assert!(buffer.len() % width == 0);
    tonemapper.initialize(&film, factor);
    buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(pixel_idx, v)| {
            let y: usize = pixel_idx / width;
            let x: usize = pixel_idx % width;
            let [r, g, b, _]: [f32; 4] = converter
                .transfer_function(tonemapper.map(&film, (x as usize, y as usize)), false)
                .into();
            *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
        });
}

pub fn power_heuristic_generic<T>(a: T, b: T) -> T
where
    T: Field + ToScalar<f32>,
{
    let w = a / (a + b);
    w
}

#[cfg(test)]
pub fn log_test_setup() {
    use simplelog::{
        ColorChoice, CombinedLogger, LevelFilter, TermLogger, TerminalMode, WriteLogger,
    };
    use std::fs::File;

    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Trace,
            simplelog::Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Trace,
            simplelog::Config::default(),
            File::create("test.log").unwrap(),
        ),
    ])
    .unwrap();
}
