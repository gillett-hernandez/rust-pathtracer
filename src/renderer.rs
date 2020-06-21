use crate::camera::{Camera, SimpleCamera};
use crate::config::RenderSettings;
use crate::integrator::Integrator;
use crate::math::*;
use crate::world::World;

use std::sync::Arc;
use std::vec::Vec;

use rayon::prelude::*;
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

pub struct NaiveRenderer {}

impl NaiveRenderer {
    pub fn new() -> NaiveRenderer {
        NaiveRenderer {}
    }
}

pub trait Renderer {
    fn render(
        &self,
        integrator: Arc<Box<dyn Integrator>>,
        camera: &Box<dyn Camera>,
        settings: &RenderSettings,
        film: &mut Film<XYZColor>,
    );
}

impl Renderer for NaiveRenderer {
    fn render(
        &self,
        integrator: Arc<Box<dyn Integrator>>,
        camera: &Box<dyn Camera>,
        settings: &RenderSettings,
        film: &mut Film<XYZColor>,
    ) {
        // for y in 0..film.height {
        //     for x in 0..film.width {
        let width = film.width;
        let height = film.height;
        film.buffer
            .par_iter_mut()
            // .iter_mut()
            .enumerate()
            .for_each(|(pixel_index, pixel_ref)| {
                let y: usize = pixel_index / width;
                let x: usize = pixel_index - width * y;
                // gen ray for pixel x, y
                // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                let mut temp_color = RGBColor::BLACK;
                // let mut temp_color = XYZColor::BLACK;
                // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                for s in 0..settings.min_samples {
                    let sample = sampler.draw_2d();
                    let r = camera.get_ray(
                        (x as f32 + sample.x) / (width as f32),
                        (y as f32 + sample.y) / (height as f32),
                    );
                    temp_color += RGBColor::from(integrator.color(&mut sampler, r));
                }
                // unsafe {
                *pixel_ref = XYZColor::from(temp_color / (settings.min_samples as f32));
                // }
            });
    }
}
