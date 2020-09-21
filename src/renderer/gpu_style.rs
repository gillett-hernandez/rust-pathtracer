use super::{output_film, Film, Renderer};

use crate::camera::{Camera, CameraId};
use crate::config::Config;
// use crate::hittable::Hittable;
use crate::integrator::gpu_style::*;
// use crate::materials::*;
use crate::math::*;
// use crate::profile::Profile;
// use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
// use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;
// use crate::MaterialId;
// use crate::world::TransportMode;

// use std::collections::HashMap;
// use std::io::Write;
// use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
// use std::cmp::Ordering;
use std::sync::Arc;
// use std::thread;
use std::time::{Duration, Instant};

// use crossbeam::channel::unbounded;
// use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

const COMPRESSED_STYLE: bool = true;
pub struct GPUStyleRenderer {}

impl GPUStyleRenderer {
    pub fn new() -> Self {
        GPUStyleRenderer {}
    }
}

impl Renderer for GPUStyleRenderer {
    fn render(&self, world: World, cameras: Vec<Camera>, config: &Config) {
        use crate::config::Resolution;

        let mut films: Vec<Film<XYZColor>> = Vec::new();
        for render_settings in config.render_settings.iter() {
            let Resolution { width, height } = render_settings.resolution;
            films.push(Film::new(width, height, XYZColor::BLACK));
        }

        let now = Instant::now();
        let arc_world = Arc::new(world.clone());
        films
            .par_iter_mut()
            .enumerate()
            .for_each(|(film_idx, mut film)| {
                let render_settings = config.render_settings[film_idx].clone();
                let integrator = GPUStylePTIntegrator::new(
                    render_settings.min_bounces.unwrap_or(0),
                    render_settings.max_bounces.unwrap_or(8),
                    arc_world.clone(),
                    render_settings.russian_roulette.unwrap_or(true),
                    render_settings.light_samples.unwrap_or(1),
                    render_settings.only_direct.unwrap_or(false),
                    render_settings
                        .wavelength_bounds
                        .map_or(Bounds1D::new(400.0, 780.0), |v| Bounds1D::new(v.0, v.1)),
                );
                let Resolution { width, height } = render_settings.resolution;
                let camera_id = render_settings.camera_id.unwrap_or(0) as CameraId;
                let kernel_width = render_settings.tile_width;
                let kernel_height = render_settings.tile_height;
                let mut primary_ray_buffer = PrimaryRayBuffer::new(kernel_width, kernel_height);
                let mut intersection_buffer = IntersectionBuffer::new(kernel_width, kernel_height);
                let mut shadow_ray_buffer = ShadowRayBuffer::new(kernel_width, kernel_height);
                let mut shading_result_buffer =
                    ShadingResultBuffer::new(kernel_width, kernel_height);
                let mut sample_buffer = SampleBounceBuffer::new(kernel_width, kernel_height);
                let x_max = (width as f32 / kernel_width as f32).floor() as usize; // change to ceil to allow partial tiles
                let y_max = (height as f32 / kernel_height as f32).floor() as usize; // change to ceil to allow partial tiles
                println!(
                    "starting tiled render with {} x {} tiles and tile size of {}x{}",
                    x_max, y_max, kernel_width, kernel_height
                );
                for y in 0..y_max {
                    for x in 0..x_max {
                        println!("{} {}", x, y);
                        let x_bounds = Bounds1D::new(
                            x as f32 / (x_max as f32),
                            (x as f32 + 1.0) / (x_max as f32),
                        );
                        let y_bounds = Bounds1D::new(
                            y as f32 / (y_max as f32),
                            (y as f32 + 1.0) / (y_max as f32),
                        );
                        let cam_bounds = Bounds2D {
                            x: x_bounds,
                            y: y_bounds,
                        };

                        if COMPRESSED_STYLE {
                            loop {
                                let status = integrator.primary_ray_pass(
                                    &mut primary_ray_buffer,
                                    &sample_buffer,
                                    kernel_width,
                                    kernel_height,
                                    render_settings
                                        .max_samples
                                        .unwrap_or(render_settings.min_samples)
                                        as usize,
                                    cam_bounds,
                                    camera_id,
                                    &cameras[camera_id as usize],
                                );
                                match status {
                                    Status::Done => {
                                        break;
                                    }
                                    _ => {}
                                }
                                integrator.intersection_pass(
                                    &primary_ray_buffer,
                                    &sample_buffer,
                                    &mut intersection_buffer,
                                );

                                integrator.nee_pass(
                                    render_settings.light_samples.unwrap() as usize,
                                    &intersection_buffer,
                                    &mut shadow_ray_buffer,
                                );
                                integrator.visibility_intersection_pass(&mut shadow_ray_buffer);
                                // intersection_buffer
                                //     .intersections
                                //     .sort_unstable_by(|a, b| intersection_cmp(&a, &b));
                                integrator.shading_pass(
                                    &intersection_buffer,
                                    &shadow_ray_buffer,
                                    cam_bounds,
                                    &mut shading_result_buffer,
                                    &mut sample_buffer,
                                    render_settings.max_bounces.unwrap_or(8) as usize,
                                    &mut primary_ray_buffer,
                                    &mut film,
                                );
                            }
                        } else {
                            // generate primary rays to fill empty spots
                            for _sample in 0..render_settings
                                .max_samples
                                .unwrap_or(render_settings.min_samples)
                            {
                                integrator.primary_ray_pass(
                                    &mut primary_ray_buffer,
                                    &sample_buffer,
                                    kernel_width,
                                    kernel_height,
                                    render_settings
                                        .max_samples
                                        .unwrap_or(render_settings.min_samples)
                                        as usize,
                                    cam_bounds,
                                    camera_id,
                                    &cameras[camera_id as usize],
                                );
                                for _ in 0..render_settings.max_bounces.unwrap() {
                                    integrator.intersection_pass(
                                        &primary_ray_buffer,
                                        &sample_buffer,
                                        &mut intersection_buffer,
                                    );

                                    integrator.nee_pass(
                                        render_settings.light_samples.unwrap() as usize,
                                        &intersection_buffer,
                                        &mut shadow_ray_buffer,
                                    );
                                    integrator.visibility_intersection_pass(&mut shadow_ray_buffer);
                                    // intersection_buffer
                                    //     .intersections
                                    //     .sort_unstable_by(|a, b| intersection_cmp(&a, &b));
                                    integrator.shading_pass(
                                        &intersection_buffer,
                                        &shadow_ray_buffer,
                                        cam_bounds,
                                        &mut shading_result_buffer,
                                        &mut sample_buffer,
                                        render_settings.max_bounces.unwrap_or(8) as usize,
                                        &mut primary_ray_buffer,
                                        &mut film,
                                    );
                                }
                            }
                        }
                        integrator.finalize_pass(
                            &mut film,
                            &sample_buffer,
                            (x * kernel_width, y * kernel_height),
                            kernel_width,
                        );
                        // sample_buffer.sample_count.fill((0, 0));
                        sample_buffer
                            .sample_count
                            .par_iter_mut()
                            .for_each(|v| *v = (0, 0));
                    }
                }
            });

        println!("{}s", now.elapsed().as_millis() as f32 / 1000.0);

        for (render_settings, film) in config.render_settings.iter().zip(films.iter()) {
            output_film(render_settings, film);
        }
    }
}
