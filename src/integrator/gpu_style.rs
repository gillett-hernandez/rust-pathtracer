use crate::renderer::Film;

use crate::camera::{Camera, CameraId};
use crate::config::Config;
use crate::hittable::Hittable;
use crate::materials::*;
use crate::math::*;
use crate::profile::Profile;
use crate::world::World;
use crate::MaterialId;
use crate::TransportMode;

use std::collections::HashMap;
// use std::io::Write;
// use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub const KERNEL_WIDTH: usize = 64;
pub const KERNEL_HEIGHT: usize = 64;
pub const KERNEL_SIZE: usize = KERNEL_HEIGHT * KERNEL_WIDTH;

#[derive(Copy, Clone, Debug)]
pub enum Pixel {
    UV(f32, f32),
    Id(usize),
}

impl Default for Pixel {
    fn default() -> Self {
        Pixel::Id(0)
    }
}

pub type CameraPixelId = Option<(CameraId, Pixel)>;

#[derive(Copy, Clone, Default, Debug)]
pub struct PrimaryRay {
    pub ray: Ray,
    pub lambda: f32,
    pub throughput: f32,
    // could be a ray from camera or ray from light.
    // ray from light will be represented with a None camera_id + pixel_uv
    pub cam_and_pixel_id: CameraPixelId,
}

pub struct PrimaryRayBuffer {
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
pub enum IntersectionData {
    Surface {
        lambda: f32,
        point: Point3,
        normal: Vec3,
        local_wi: Vec3,
        local_wo: Option<Vec3>,
        instance_id: usize,
        throughput: f32,
        uv: (f32, f32),
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

pub struct IntersectionBuffer {
    pub intersections: Vec<IntersectionData>,
}

impl IntersectionBuffer {
    pub fn new() -> Self {
        IntersectionBuffer {
            intersections: vec![IntersectionData::Empty; KERNEL_SIZE],
        }
    }
}

pub struct ShadowRayBuffer {
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

pub struct ShadingResultBuffer {
    pub data: Vec<(SingleEnergy, Vec3)>,
}

impl ShadingResultBuffer {
    pub fn new() -> Self {
        ShadingResultBuffer {
            data: vec![(SingleEnergy::ZERO, Vec3::Z); KERNEL_SIZE],
        }
    }
}

pub struct SampleCountBuffer {
    pub sample_count: Vec<usize>,
}

impl SampleCountBuffer {
    pub fn new(samples: usize) -> Self {
        SampleCountBuffer {
            sample_count: vec![samples; KERNEL_SIZE],
        }
    }
}

pub fn intersection_cmp(a: &IntersectionData, b: &IntersectionData) -> Ordering {
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

pub struct GPUStylePTIntegrator {
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
        sample_buffer: &mut SampleCountBuffer,
        width: usize,
        height: usize,
        bounds: Bounds2D,
        cam_id: CameraId,
        camera: &Camera,
    ) {
        // println!("cam window is {:?} {:?}", bounds.x, bounds.y);
        buffer.rays.par_iter_mut().enumerate().for_each(|(idx, b)| {
            // idx ranges from 0 to KERNEL_SIZE
            let tile_u = (idx % width) as f32 / width as f32; // + pixel_offset.0;
            let tile_v = (idx / width) as f32 / height as f32; // + pixel_offset.1;
            let tile_width = bounds.x.span();
            let tile_height = bounds.y.span();
            let (px, py) = (
                bounds.x.lower + (tile_u as f32) * tile_width,
                bounds.y.lower + (tile_v as f32) * tile_height,
            );
            // println!(
            //     "pixel x and y were {} {}, but uv is now {:?} {:?}",
            //     pixel_x, pixel_y, px, py
            // );
            let ray = camera.get_ray(Sample2D::new_random_sample(), px, py);
            *b = Some(PrimaryRay {
                ray,
                lambda: self.wavelength_bounds.lower
                    + self.wavelength_bounds.span() * Sample1D::new_random_sample().x,
                throughput: 1.0,
                cam_and_pixel_id: Some((cam_id, Pixel::UV(px, py))),
            });
        });
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
                if let Some(primary) = primary {
                    let maybe_hit = self.world.hit(primary.ray, 0.0, INFINITY);
                    match maybe_hit {
                        Some(hit) => {
                            let frame = TangentFrame::from_normal(hit.normal);
                            let wi = primary.ray.origin - hit.point;
                            *isect = IntersectionData::Surface {
                                lambda: primary.lambda,
                                local_wi: frame.to_local(&wi),
                                point: hit.point,
                                normal: hit.normal,
                                instance_id: hit.instance_id,
                                throughput: primary.throughput,
                                transport_mode: hit.transport_mode,
                                local_wo: None,
                                material: hit.material,
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
                }
            });
    }
    pub fn visibility_intersection_pass(&self, buffer: &mut ShadowRayBuffer) {
        // actually performs intersection with scene, putting results in the intersection buffer
        buffer.rays.par_iter_mut().for_each(|shadow| {
            let ray = shadow.0;
            let maybe_hit = self.world.hit(ray, 0.0, INFINITY);
            match maybe_hit {
                Some(_hit) => {
                    shadow.1 = true;
                }
                None => {
                    // handle no hit case
                    shadow.1 = false;
                }
            }
        });
    }
    pub fn nee_pass(
        &self,
        intersection_buffer: &IntersectionBuffer,
        shadow_buffer: &mut ShadowRayBuffer,
    ) {
        // pick a single light to try and sample?
        let (light, light_pick_pdf) = self
            .world
            .pick_random_light(Sample1D::new_random_sample())
            .unwrap();
        // queues up NEE shadow rays
        shadow_buffer
            .rays
            .par_iter_mut()
            .zip(intersection_buffer.intersections.par_iter())
            .for_each(|((ray, _hit), isect)| {
                if let IntersectionData::Surface { point, .. } = isect {
                    let (wo, _pdf) = light.sample(Sample2D::new_random_sample(), *point);
                    *ray = Ray::new(*point, wo);
                } else {
                }
            });
    }
    pub fn shading_pass(
        &self,
        intersection_buffer: &IntersectionBuffer,
        shadow_buffer: &ShadowRayBuffer,
        cam_bounds: Bounds2D,
        buffer: &mut ShadingResultBuffer,
        rays: &mut PrimaryRayBuffer,
        film: &mut Film<XYZColor>,
    ) {
        // performs the scatter, pdf, and shading calculations, putting the results in the shading result buffer and back into the primary ray buffer (for bounces)

        // perform attenuation and scatter
        buffer
            .data
            .par_iter_mut()
            .zip(intersection_buffer.intersections.par_iter())
            .enumerate()
            .for_each(|(index, (shading_result, isect_data))| {
                match isect_data {
                    IntersectionData::Surface {
                        local_wi,
                        uv,
                        // normal,
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

        // generate new rays based on if the interaction was one that bounced, based on the isect data
        // cannot be parallelized because we're also modifying the film at the same time
        for (i, ray) in rays.rays.iter_mut().enumerate() {
            if ray.is_none() {
                continue;
            }
            let isect = intersection_buffer.intersections[i];
            let (throughput, wo) = buffer.data[i];
            /*
            let tile_u = (idx % width) as f32 / width as f32; // + pixel_offset.0;
            let tile_v = (idx / width) as f32 / height as f32; // + pixel_offset.1;
            let tile_width = bounds.x.span();
            let tile_height = bounds.y.span();
            let (px, py) = (
                bounds.x.lower + (tile_u as f32) * tile_width,
                bounds.y.lower + (tile_v as f32) * tile_height,
            );
            */
            let cam_and_pixel_id = ray.unwrap().cam_and_pixel_id;
            if let Some((cam_id, pixel)) = cam_and_pixel_id {
                let lambda = ray.unwrap().lambda;
                let idx = match pixel {
                    Pixel::Id(id) => id,
                    Pixel::UV(u, v) => {
                        (u * film.width as f32) as usize
                            + ((v * film.height as f32) as usize * film.width)
                    }
                };
                if let Some(v) = film.buffer.get_mut(idx) {
                    *v += XYZColor::from(SingleWavelength::new(lambda, throughput));
                }
            }

            if let IntersectionData::Surface { point, lambda, .. } = isect {
                *ray = Some(PrimaryRay {
                    ray: Ray::new(point, wo),
                    lambda,
                    throughput: throughput.0,
                    cam_and_pixel_id,
                });
            } else {
                *ray = None;
            }
        }
    }
}
