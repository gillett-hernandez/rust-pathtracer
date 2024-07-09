#![feature(portable_simd)]

#[macro_use]
extern crate smallvec;
#[macro_use]
extern crate tracing;
#[macro_use]
extern crate paste;

#[cfg(feature = "minifb")]
use minifb::{Key, Window, WindowOptions};
#[cfg(feature = "minifb")]
use rayon::prelude::*;
#[cfg(feature = "minifb")]
use tonemap::{sRGB, Color, Tonemapper, OETF};
#[cfg(feature = "minifb")]
use vec2d::Vec2D;

use math::{
    prelude::XYZColor,
    traits::{Field, ToScalar},
};

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
pub mod prelude;
pub mod profile;
pub mod renderer;
pub mod texture;
pub mod tonemap;
pub mod vec2d;
pub mod world;

#[cfg(test)]
pub mod props;

// mauve. universal sign of danger
pub const MAUVE: XYZColor = XYZColor::new(0.5199467, 51.48687, 1.0180528);

pub const NORMAL_OFFSET: f32 = 0.001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub enum TransportMode {
    Radiance,
    #[default]
    Importance,
}

pub fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[cfg(feature = "minifb")]
pub fn window_loop<F>(
    width: usize,
    height: usize,
    max_framerate: usize,
    options: WindowOptions,
    clear_buffer: bool,
    mut func: F,
) where
    F: FnMut(&Window, &mut Vec<u32>, usize, usize) -> (),
{
    let mut window = Window::new("Window", width, height, options).unwrap();
    window.set_target_fps(max_framerate);

    let mut film = Vec2D::new(width, height, 0u32);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if clear_buffer {
            film.buffer.fill(0u32);
        }
        func(&window, &mut film.buffer, width, height);

        window
            .update_with_buffer(&film.buffer, width, height)
            .unwrap();
    }
}

#[cfg(feature = "minifb")]
pub fn update_window_buffer(
    buffer: &mut [u32],
    film: &Vec2D<XYZColor>,
    tonemapper: &mut dyn Tonemapper,
    factor: f32,
) {
    use crate::tonemap::{Rec709Primaries, CIEXYZ};

    let width = film.width;
    debug_assert!(buffer.len() % width == 0);
    tonemapper.initialize(&film, factor);
    buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(pixel_idx, v)| {
            let y: usize = pixel_idx / width;
            let x: usize = pixel_idx % width;
            let as_xyz: Color<CIEXYZ> = tonemapper.map(&film, (x as usize, y as usize)).into();
            let as_rec709: Color<Rec709Primaries> = as_xyz.into();
            let [r, g, b, _]: [f32; 4] = sRGB::oetf(as_rec709.values).into();
            *v = rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
        });
}

pub fn power_heuristic_generic<T>(a: T, b: T) -> T
where
    T: Field + ToScalar<f32>,
{
    a / (a + b)
}

#[cfg(test)]
pub fn log_test_setup() {
    // use simplelog::{
    //     ColorChoice, CombinedLogger, LevelFilter, TermLogger, TerminalMode, WriteLogger,
    // };
    // use std::fs::File;

    // CombinedLogger::init(vec![
    //     TermLogger::new(
    //         LevelFilter::Trace,
    //         simplelog::Config::default(),
    //         TerminalMode::Mixed,
    //         ColorChoice::Auto,
    //     ),
    //     WriteLogger::new(
    //         LevelFilter::Trace,
    //         simplelog::Config::default(),
    //         File::create("test.log").unwrap(),
    //     ),
    // ])
    // .unwrap();

    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(Level::TRACE)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}
