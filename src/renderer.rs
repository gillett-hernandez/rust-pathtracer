use std::vec::Vec;

use crate::camera::{Camera, SimpleCamera};
use crate::config::RenderSettings;
use crate::integrator::Integrator;
use crate::math::*;
use crate::world::World;
pub struct Film<T> {
    pub buffer: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T: Copy> Film<T> {
    pub fn new(width: usize, height: usize, fill_value: T) -> Film<T> {
        // allocate with
        let capacity: usize = (width * height) as usize;
        let mut buffer: Vec<T> = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            buffer.push(fill_value);
        }
        Film {
            buffer,
            width,
            height,
        }
    }
}

pub struct NaiveRenderer {
    integrator: Box<dyn Integrator>,
}

impl NaiveRenderer {
    pub fn new(integrator: Box<dyn Integrator>) -> NaiveRenderer {
        NaiveRenderer { integrator }
    }
}

pub trait Renderer {
    fn render(&self, film: &mut Film<RGBColor>, camera: &Box<dyn Camera>, config: &RenderSettings);
}

impl Renderer for NaiveRenderer {
    fn render(&self, film: &mut Film<RGBColor>, camera: &Box<dyn Camera>, config: &RenderSettings) {
        for y in 0..film.height {
            for x in 0..film.width {
                // gen ray for pixel x, y
                // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                for s in 0..config.min_samples.unwrap_or(1) {
                    let r = camera.get_ray(
                        (x as f32 + random()) / (film.width as f32),
                        (y as f32 + random()) / (film.height as f32),
                    );
                    let color = self.integrator.color(r);
                    film.buffer[y * film.width + x] += color;
                }
            }
        }
    }
}
