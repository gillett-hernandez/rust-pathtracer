use minifb::{Key, Window, WindowOptions};

pub use packed_simd::f32x4;

pub use crate::camera::{Camera, CameraEnum, CameraId};
pub use crate::curves::*;
pub use crate::materials::{Material, MaterialEnum, MaterialId};
pub use crate::renderer::Film;
pub use crate::texture::TexStack;

pub use math::spectral::{BOUNDED_VISIBLE_RANGE, EXTENDED_VISIBLE_RANGE};
pub use math::*;

pub use std::f32::consts::{PI, SQRT_2, TAU};

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
