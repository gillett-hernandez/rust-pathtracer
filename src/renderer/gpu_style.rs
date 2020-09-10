use super::{output_film, Film, Renderer};

use crate::camera::{Camera, CameraId};
use crate::config::Config;
use crate::materials::*;
use crate::math::*;
use crate::profile::Profile;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;
use crate::MaterialId;
use crate::TransportMode;

use std::collections::HashMap;
// use std::io::Write;
// use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::cmp::Ordering;
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

type CameraPixelId = Option<(CameraId, f32, f32)>;

#[derive(Copy, Clone, Default, Debug)]
struct PrimaryRay {
    pub ray: Ray,
    pub lambda: f32,
    pub throughput: f32,
    // could be a ray from camera or ray from light.
    // ray from light will be represented with a None camera_id + pixel_uv
    pub cam_and_pixel_id: CameraPixelId,
}

struct PrimaryRayBuffer {
    pub rays: Vec<Option<PrimaryRay>>,
}

impl PrimaryRayBuffer {
    pub fn new() -> Self {
        PrimaryRayBuffer {
            rays: vec![None; KERNEL_SIZE],
        }
    }
}

#[derive(Copy, Clone)]
enum IntersectionData {
    Surface {
        lambda: f32,
        point: Point3,
        normal: Vec3,
        local_wi: Vec3,
        local_wo: Option<Vec3>,
        instance_id: usize,
        throughput: f32,
        uv: (f32, f32),
        pdf: f32,
        camera_pixel_id: CameraPixelId,
        material: MaterialId,
        transport_mode: TransportMode,
    },
    Environment {
        lambda: f32,
        uv: (f32, f32),
    },
    Empty,
}

struct IntersectionBuffer {
    pub intersections: Vec<IntersectionData>,
}

impl IntersectionBuffer {
    pub fn new() -> Self {
        IntersectionBuffer {
            intersections: vec![IntersectionData::Empty; KERNEL_SIZE],
        }
    }
}

struct ShadowRayBuffer {
    // Ray, and whether it intersected anything. defaults to true, set to false upon tracing and testing.
    pub rays: Vec<(Ray, bool)>,
}

impl ShadowRayBuffer {
    pub fn new() -> Self {
        ShadowRayBuffer {
            rays: vec![(Ray::default(), true); KERNEL_SIZE],
        }
    }
}

struct ShadingResultBuffer {
    pub data: Vec<(SingleEnergy, Vec3)>,
}

impl ShadingResultBuffer {
    pub fn new() -> Self {
        ShadingResultBuffer {
            data: vec![(SingleEnergy::ZERO, Vec3::ZERO); KERNEL_SIZE],
        }
    }
}

struct SampleCountBuffer {
    pub sample_count: Vec<usize>,
}

impl SampleCountBuffer {
    pub fn new(samples: usize) -> Self {
        SampleCountBuffer {
            sample_count: vec![samples; KERNEL_SIZE],
        }
    }
}

fn intersection_cmp(a: &IntersectionData, b: &IntersectionData) -> Ordering {
    if let (
        IntersectionData::Surface {
            material: mat_a,
            uv: uv_a,
            ..
        },
        IntersectionData::Surface {
            material: mat_b,
            uv: uv_b,
            ..
        },
    ) = (a, b)
    {
        match (mat_a, mat_b) {
            (_, MaterialId::Camera(_)) => Ordering::Less,
            (MaterialId::Camera(_), _) => Ordering::Greater,
            (MaterialId::Light(_), MaterialId::Material(_)) => Ordering::Less,
            (MaterialId::Material(_), MaterialId::Light(_)) => Ordering::Greater,
            (MaterialId::Light(a), MaterialId::Light(b)) => {
                let ord = a.partial_cmp(&b).unwrap();
                if ord != Ordering::Equal {
                    ord
                } else {
                    if let Some(y_cmp) = uv_a.1.partial_cmp(&uv_b.1) {
                        match y_cmp {
                            Ordering::Equal => {
                                if let Some(x_cmp) = uv_a.0.partial_cmp(&uv_b.0) {
                                    x_cmp
                                } else {
                                    panic!();
                                }
                            }
                            _ => y_cmp,
                        }
                    } else {
                        panic!();
                    }
                }
            }
            (MaterialId::Material(a), MaterialId::Material(b)) => {
                let ord = a.partial_cmp(&b).unwrap();
                if ord != Ordering::Equal {
                    ord
                } else {
                    if let Some(y_cmp) = uv_a.1.partial_cmp(&uv_b.1) {
                        match y_cmp {
                            Ordering::Equal => {
                                if let Some(x_cmp) = uv_a.0.partial_cmp(&uv_b.0) {
                                    x_cmp
                                } else {
                                    panic!();
                                }
                            }
                            _ => y_cmp,
                        }
                    } else {
                        panic!();
                    }
                }
            }
        }
    } else {
        // TODO: put env hit sorting here
        Ordering::Equal
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
            *b = Some(PrimaryRay {
                ray,
                lambda: self.wavelength_bounds.lower
                    + self.wavelength_bounds.span() * Sample1D::new_random_sample().x,
                throughput: 1.0,
                cam_and_pixel_id: Some((cam_id, px, py)),
            });
        });
    }
    pub fn shadow_ray_pass(
        &self,
        intersetion_buffer: &IntersectionBuffer,
        buffer: &mut ShadowRayBuffer,
    ) {
        // checks visibility
    }
    pub fn intersection_pass(
        &self,
        ray_buffer: &PrimaryRayBuffer,
        buffer: &mut IntersectionBuffer,
    ) {
        // actually performs intersection with scene, putting results in the intersection buffer
        buffer
            .intersections
            .par_iter_mut()
            .zip(ray_buffer.rays.par_iter())
            .for_each(|(isect, primary)| {
                let primary = primary.unwrap();
                let maybe_hit = self.world.hit(primary.ray, 0.0, INFINITY);
                match maybe_hit {
                    Some(hit) => {
                        let frame = TangentFrame::from_normal(hit.normal);
                        let wi = primary.ray.origin - hit.point;
                        *isect = IntersectionData::Surface {
                            lambda: hit.lambda,
                            local_wi: frame.to_local(&wi),
                            point: hit.point,
                            normal: hit.normal,
                            instance_id: hit.instance_id,
                            throughput: primary.throughput,
                            transport_mode: hit.transport_mode,
                            local_wo: None,
                            material: hit.material,
                            pdf: 0.0,
                            uv: hit.uv,
                            camera_pixel_id: primary.cam_and_pixel_id,
                        }
                    }
                    None => {
                        // handle env hit
                        *isect = IntersectionData::Environment {
                            uv: direction_to_uv(primary.ray.direction),
                            lambda: primary.lambda,
                        }
                    }
                }
            });
    }
    pub fn shading_pass(
        &self,
        intersection_buffer: &IntersectionBuffer,
        buffer: &mut ShadingResultBuffer,
        rays: &mut PrimaryRayBuffer,
    ) {
        // performs the scatter, pdf, and shading calculations, putting the results in the shading result buffer and back into the primary ray buffer (for bounces)

        buffer
            .data
            .par_iter_mut()
            .zip(intersection_buffer.intersections.par_iter())
            .enumerate()
            .for_each(|(index, (shading_result, isect_data))| {
                match isect_data {
                    IntersectionData::Surface {
                        local_wi,
                        mut local_wo,
                        uv,
                        normal,
                        lambda,
                        throughput,
                        material,
                        transport_mode,
                        ..
                    } => {
                        let material = self.world.get_material(*material);
                        let local_wo = material.generate(
                            *lambda,
                            *uv,
                            *transport_mode,
                            Sample2D::new_random_sample(),
                            *local_wi,
                        );
                        let f = local_wo.map_or(SingleEnergy::ZERO, |v| {
                            material.f(*lambda, *uv, *transport_mode, *local_wi, v)
                        });
                        let pdf = local_wo.map_or(0.0.into(), |v| {
                            material.scatter_pdf(*lambda, *uv, *transport_mode, *local_wi, v)
                        });
                        *shading_result = (*throughput * f / pdf.0, local_wo.unwrap_or(Vec3::ZERO));
                    }
                    IntersectionData::Environment { uv, lambda, .. } => {
                        let energy = self.world.environment.emission(*uv, *lambda);
                        // shading_result.
                        *shading_result = (energy, Vec3::ZERO);
                    }
                    _ => {}
                }
            });
        for (i, ray) in rays.rays.iter_mut().enumerate() {
            let isect = intersection_buffer.intersections[i];
            let (energy, wo) = buffer.data[i];
            let lambda = ray.unwrap().lambda;
            let cam_and_pixel_id = ray.unwrap().cam_and_pixel_id;
            if let IntersectionData::Surface { point, lambda, .. } = isect {
                *ray = Some(PrimaryRay {
                    ray: Ray::new(point, wo),
                    lambda,
                    throughput: energy.0,
                    cam_and_pixel_id,
                });
            } else {
                *ray = None;
            }
        }
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
        for render_settings in config.render_settings.iter() {
            let Resolution { width, height } = render_settings.resolution;
            films.push(Film::new(width, height, XYZColor::BLACK));
        }
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
                let mut primary_ray_buffer = PrimaryRayBuffer::new();
                let mut intersection_buffer = IntersectionBuffer::new();
                let mut shadow_ray_buffer = ShadowRayBuffer::new();
                let mut shading_result_buffer = ShadingResultBuffer::new();
                let mut sample_buffer =
                    SampleCountBuffer::new(render_settings.min_samples as usize);
                let x_max = (width as f32 / KERNEL_WIDTH as f32).ceil() as usize;
                let y_max = (height as f32 / KERNEL_HEIGHT as f32).ceil() as usize;
                for y in 0..y_max {
                    for x in 0..x_max {
                        println!("{} {}", x, y);
                        let x_bounds = Bounds1D::new(x as f32 / 5.0, (x as f32 + 1.0) / 5.0);
                        let y_bounds = Bounds1D::new(y as f32 / 5.0, (y as f32 + 1.0) / 5.0);
                        integrator.primary_ray_pass(
                            &mut primary_ray_buffer,
                            width,
                            Bounds2D {
                                x: x_bounds,
                                y: y_bounds,
                            },
                            camera_id,
                            &cameras[camera_id as usize],
                        );
                        integrator.intersection_pass(&primary_ray_buffer, &mut intersection_buffer);
                        integrator.shadow_ray_pass(&intersection_buffer, &mut shadow_ray_buffer);
                        intersection_buffer
                            .intersections
                            .sort_unstable_by(|a, b| intersection_cmp(&a, &b));
                        integrator.shading_pass(
                            &intersection_buffer,
                            &mut shading_result_buffer,
                            &mut primary_ray_buffer,
                        );
                        // integrator.nee_pass(&intersection_buffer, &shadow_ray_buffer, &mut film);
                    }
                }
            });
        for (render_settings, film) in config.render_settings.iter().zip(films.iter()) {
            output_film(render_settings, film);
        }
    }
}
