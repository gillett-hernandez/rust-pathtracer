use crate::camera::Camera;
use crate::config::RenderSettings;
use crate::integrator::Integrator;
use crate::math::*;

use std::io::Write;
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
    pub fn at(&self, x: usize, y: usize) -> T {
        self.buffer[y * self.width + x]
    }

    pub fn total_pixels(&self) -> usize {
        self.width * self.height
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

        for _ in 0..100 {
            print!("-");
        }
        println!("");
        let output_divisor = (film.width * film.height / 100).max(1);
        film.buffer
            .par_iter_mut()
            // .iter_mut()
            .enumerate()
            .for_each(|(pixel_index, pixel_ref)| {
                let y: usize = pixel_index / width;
                let x: usize = pixel_index - width * y;
                // gen ray for pixel x, y
                // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                // let mut temp_color = RGBColor::BLACK;
                let mut temp_color = XYZColor::BLACK;
                let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.

                for _s in 0..settings.min_samples {
                    let sample = sampler.draw_2d();
                    let r = camera.get_ray(
                        (x as f32 + sample.x) / (width as f32),
                        (y as f32 + sample.y) / (height as f32),
                    );
                    temp_color += XYZColor::from(integrator.color(&mut sampler, r));
                    // temp_color += RGBColor::from(integrator.color(&mut sampler, r));
                    assert!(
                        temp_color.0.is_finite().all(),
                        "{:?} resulted in {:?}",
                        r,
                        temp_color
                    );
                }
                if pixel_index % output_divisor == 0 {
                    let stdout = std::io::stdout();
                    let mut handle = stdout.lock();
                    handle.write_all(b".").unwrap();
                    std::io::stdout().flush().expect("some error message")
                }
                // unsafe {
                *pixel_ref = temp_color / (settings.min_samples as f32);
                // }
            });
    }
}
