use super::{output_film, Film, Renderer};

use crate::camera::{Camera, CameraId};
use crate::config::Config;
use crate::math::*;
use crate::profile::Profile;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;
use crate::MaterialId;

use std::collections::HashMap;
// use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::unbounded;
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

const KERNEL_WIDTH: usize = 256;
const KERNEL_HEIGHT: usize = 256;
const KERNEL_SIZE: usize = KERNEL_HEIGHT * KERNEL_WIDTH;

#[derive(Copy, Clone, Default, Debug)]
struct PrimaryRay {
    pub ray: Ray,
    // could be a ray from camera or ray from light.
    // ray from light will be represented with a None camera_id + pixel_uv
    pub cam_and_pixel_id: Option<(CameraId, f32, f32)>,
}

struct PrimaryRayBuffer {
    pub rays: [PrimaryRay; KERNEL_SIZE],
}

impl PrimaryRayBuffer {
    pub fn new() -> Self {
        PrimaryRayBuffer {
            rays: [PrimaryRay::default(); KERNEL_SIZE],
        }
    }
}

#[derive(Default, Copy, Clone)]
struct IntersectionData {
    pub point: Point3,
    pub normal: Vec3,
    pub local_wi: Vec3,
    pub local_wo: Option<Vec3>,
    pub uv: (f32, f32),
    pub pdf: f32,
    pub material: MaterialId,
}

struct IntersectionBuffer {
    pub intersections: [IntersectionData; KERNEL_SIZE],
}

impl IntersectionBuffer {
    pub fn new() -> Self {
        IntersectionBuffer {
            intersections: [IntersectionData::default(); KERNEL_SIZE],
        }
    }
}

struct ShadowRayBuffer {
    // Ray, and whether it intersected anything. defaults to true, set to false upon tracing and testing.
    pub rays: [(Ray, bool); KERNEL_SIZE],
}

impl ShadowRayBuffer {
    pub fn new() -> Self {
        ShadowRayBuffer {
            rays: [(Ray::default(), true); KERNEL_SIZE],
        }
    }
}

struct ShadingResultBuffer {
    pub data: [(SingleEnergy, Vec3); KERNEL_SIZE],
}

impl ShadingResultBuffer {
    pub fn new() -> Self {
        ShadingResultBuffer {
            data: [(SingleEnergy::ZERO, Vec3::ZERO); KERNEL_SIZE],
        }
    }
}

struct SampleCountBuffer {
    pub sample_count: [usize; KERNEL_SIZE],
}

impl SampleCountBuffer {
    pub fn new(samples: usize) -> Self {
        SampleCountBuffer {
            sample_count: [samples; KERNEL_SIZE],
        }
    }
}

struct GPUStylePTIntegrator {
    pub min_bounces: u16,
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub light_samples: u16,
    pub only_direct: bool,
    pub wavelength_bounds: Bounds1D,
}

impl GPUStylePTIntegrator {
    pub fn new(
        min_bounces: u16,
        max_bounces: u16,
        world: Arc<World>,
        russian_roulette: bool,
        light_samples: u16,
        only_direct: bool,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        GPUStylePTIntegrator {
            min_bounces,
            max_bounces,
            world,
            russian_roulette,
            light_samples,
            only_direct,
            wavelength_bounds,
        }
    }
    pub fn primary_ray_pass(
        &self,
        buffer: &mut PrimaryRayBuffer,
        width: usize,
        bounds: Bounds2D,
        cam_id: CameraId,
        camera: &Camera,
    ) {
        buffer.rays.par_iter_mut().enumerate().for_each(|(a, b)| {
            let pixel_x = a % width;
            let pixel_y = a / width;
            let bounds_x_span = bounds.x.span();
            let bounds_y_span = bounds.y.span();
            let (px, py) = (
                bounds.x.lower + (pixel_x as f32) / bounds_x_span,
                bounds.y.lower + (pixel_y as f32) / bounds_y_span,
            );
            let ray = camera.get_ray(Sample2D::new_random_sample(), px, py);
            *b = PrimaryRay {
                ray,
                cam_and_pixel_id: Some((cam_id, px, py)),
            }
        });
    }
    pub fn bounce_ray_pass(&self, buffer: &mut PrimaryRayBuffer) {}
    pub fn shadow_ray_pass(&self, buffer: &mut ShadowRayBuffer) {}
    pub fn intersection_pass(&self, rays: &PrimaryRayBuffer, buffer: &mut IntersectionBuffer) {}
    pub fn shading_pass(
        &self,
        intersection_buffer: &IntersectionBuffer,
        buffer: &mut ShadingResultBuffer,
    ) {
    }
}

pub struct GPUStyleRenderer {}

impl GPUStyleRenderer {
    pub fn new() -> Self {
        GPUStyleRenderer {}
    }
}

impl Renderer for GPUStyleRenderer {
    fn render(&self, mut world: World, cameras: Vec<Camera>, config: &Config) {
        use crate::config::Resolution;

        let mut films: Vec<Film<XYZColor>> = Vec::new();
        let arc_world = Arc::new(world.clone());
        films
            .par_iter_mut()
            .enumerate()
            .for_each(|(film_idx, film)| {
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
                let mut primary_ray_buffer = PrimaryRayBuffer::new();
                let mut intersection_buffer = IntersectionBuffer::new();
                let mut shadow_ray_buffer = ShadowRayBuffer::new();
                let mut shading_result_buffer = ShadingResultBuffer::new();
                for _ in 0..render_settings.min_samples {
                    integrator.primary_ray_pass(
                        &mut primary_ray_buffer,
                        width,
                        Bounds2D {
                            x: Bounds1D::new(0.0, 1.0),
                            y: Bounds1D::new(0.0, 1.0),
                        },
                        camera_id,
                        &cameras[camera_id as usize],
                    );
                    integrator.intersection_pass(&primary_ray_buffer, &mut intersection_buffer);
                    integrator.shading_pass(&intersection_buffer, &mut shading_result_buffer);
                }
            });
        for (render_settings, film) in config.render_settings.iter().zip(films.iter()) {
            output_film(render_settings, film);
        }
    }
}
