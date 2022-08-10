#![feature(result_option_inspect, backtrace)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate packed_simd;
#[macro_use]
extern crate paste;

extern crate image;

extern crate minifb;

pub use math;
use math::XYZColor;
use minifb::{Window, WindowOptions, Key};
use renderer::Film;

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

// mauve. universal sign of danger
pub const MAUVE: XYZColor = XYZColor::new(0.5199467, 51.48687, 1.0180528);


pub fn window_loop<F>(width: usize, height: usize, max_framerate: usize, options: WindowOptions, mut func: F)
where
    F: FnMut(&Window, &mut Vec<u32>, usize, usize) -> (),
{
    let mut window = Window::new(
        "Window",
        width,
        height,
        options,
    )
    .unwrap();
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
