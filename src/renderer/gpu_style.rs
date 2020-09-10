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

#[derive(Copy, Clone, Default, Debug)]
struct PrimaryRay {
    pub ray: Ray,
    // could be a ray from camera or ray from light.
    // ray from light will be represented with a None camera_id + pixel_uv
    pub cam_and_pixel_id: Option<(CameraId, f32, f32)>,
}

struct PrimaryRayQueue {
    pub rays: Vec<PrimaryRay>,
}

impl PrimaryRayQueue {
    pub fn new(count: usize) -> Self {
        let mut vec = Vec::new();
        vec.resize(count, PrimaryRay::default());
        PrimaryRayQueue { rays: vec }
    }
}

struct BounceRayQueue {
    pub rays: Vec<PrimaryRay>,
}

struct ShadowRayQueue {
    // Ray, and whether it intersected anything. defaults to true, set to false upon tracing and testing.
    pub rays: Vec<(Ray, bool)>,
}

struct ShadingRequestQueue {
    // each entry in the shading request queue contains all the data needed to actually shade a hit point
    pub shading_requests: Vec<(f32, f32, Vec3, Option<Vec3>, MaterialId)>,
    pub shading_results: Vec<(SingleEnergy, Vec3)>,
}

struct SampleCountBuffer {
    pub sample_count: Vec<usize>,
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
        buffer: &mut PrimaryRayQueue,
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
    pub fn bounce_ray_pass(&self, buffer: &mut BounceRayQueue) {}
    pub fn shadow_ray_pass(&self, buffer: &mut ShadowRayQueue) {}
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
                let camera_id = render_settings.camera_id.unwrap_or(0) as u8;
                let mut buffer = PrimaryRayQueue::new(width * height);
                integrator.primary_ray_pass(
                    &mut buffer,
                    width,
                    Bounds2D {
                        x: Bounds1D::new(0.0, 1.0),
                        y: Bounds1D::new(0.0, 1.0),
                    },
                    camera_id,
                    &cameras[camera_id as usize],
                );
            });
        for (render_settings, film) in config.render_settings.iter().zip(films.iter()) {
            output_film(render_settings, film);
        }
    }
}
