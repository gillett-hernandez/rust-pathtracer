use crate::renderer::Film;

use crate::camera::{Camera, CameraId};
// use crate::config::Config;
use crate::hittable::Hittable;
use crate::materials::*;
use crate::math::*;
// use crate::profile::Profile;
use crate::world::World;
use crate::MaterialId;
use crate::TransportMode;

// use std::collections::HashMap;
// use std::io::Write;
// use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub const KERNEL_WIDTH: usize = 128;
pub const KERNEL_HEIGHT: usize = 128;
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
    pub buffer_idx: usize,
    pub transport_mode: TransportMode,
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

#[derive(Copy, Clone, Debug)]
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
        emission: SingleEnergy,
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

#[derive(Copy, Clone, Debug)]
pub enum ConnectionData {
    Intersection(IntersectionData),
}

#[derive(Copy, Clone, Debug)]
pub enum VisibilityTestStatus {
    Unchecked,
    Visible,
    Blocked,
}

pub struct ShadowRayBuffer {
    // Ray, and whether it intersected anything. defaults to true, set to false upon tracing and testing.
    pub rays: Vec<Option<(PrimaryRay, VisibilityTestStatus, Option<ConnectionData>)>>,
}

impl ShadowRayBuffer {
    pub fn new() -> Self {
        ShadowRayBuffer {
            rays: vec![None; KERNEL_SIZE],
        }
    }
}

pub struct ShadingResultBuffer {
    pub data: Vec<(SingleEnergy, SingleEnergy, Option<Vec3>)>,
}

impl ShadingResultBuffer {
    pub fn new() -> Self {
        ShadingResultBuffer {
            data: vec![(SingleEnergy::ZERO, SingleEnergy::ZERO, None); KERNEL_SIZE],
        }
    }
}

pub struct SampleBounceBuffer {
    pub sample_count: Vec<(usize, usize)>,
}

impl SampleBounceBuffer {
    pub fn new() -> Self {
        SampleBounceBuffer {
            sample_count: vec![(0, 0); KERNEL_SIZE],
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

#[derive(Debug, Copy, Clone)]
pub enum Status {
    Running,
    Underutilized,
    Done,
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
        sample_buffer: &mut SampleBounceBuffer,
        width: usize,
        height: usize,
        max_samples: usize,
        bounds: Bounds2D,
        cam_id: CameraId,
        camera: &Camera,
    ) -> Status {
        // buffer.rays.par_sort_unstable_by(|a, b| match (a, b) {
        //     (Some(_), None) => Ordering::Less,
        //     (None, Some(_)) => Ordering::Greater,
        //     _ => Ordering::Equal,
        // });
        // let first_empty = buffer.rays.partition_point(|a| a.is_some());
        // println!("first empty is {}", first_empty);
        let any_remaining = buffer
            .rays
            .par_iter_mut()
            .zip(sample_buffer.sample_count.par_iter())
            .enumerate()
            .map(|(index, (primary_ray, (sample_count, bounce_count)))| {
                if *bounce_count == 0 && *sample_count < max_samples {
                    // ready to fire from camera
                    let tile_u = (index % width) as f32 / width as f32; // + pixel_offset.0;
                    let tile_v = (index / width) as f32 / height as f32; // + pixel_offset.1;
                    let tile_width = bounds.x.span();
                    let tile_height = bounds.y.span();
                    let (px, py) = (
                        bounds.x.lower + (tile_u as f32) * tile_width,
                        bounds.y.lower + (tile_v as f32) * tile_height,
                    );
                    let ray = camera.get_ray(Sample2D::new_random_sample(), px, py);
                    *primary_ray = Some(PrimaryRay {
                        ray,
                        lambda: self.wavelength_bounds.lower
                            + self.wavelength_bounds.span() * Sample1D::new_random_sample().x,
                        buffer_idx: index,
                        throughput: 1.0,
                        transport_mode: TransportMode::default(),
                        cam_and_pixel_id: Some((cam_id, Pixel::UV(px, py))),
                    });
                    true
                } else {
                    if *sample_count < max_samples {
                        true
                    } else {
                        false
                    }
                }
            })
            .reduce(|| false, |a, b| a || b);
        if any_remaining {
            Status::Running
        } else {
            Status::Done
        }
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
                    // perform russian roulette
                    let mut mult = 1.0;
                    if primary.throughput < 0.05 {
                        let x = Sample1D::new_random_sample().x;
                        if x < primary.throughput {
                            mult *= 1.0 / x;
                        } else {
                            *isect = IntersectionData::Empty;
                            return;
                        }
                    }
                    let maybe_hit = self.world.hit(primary.ray, 0.0001, primary.ray.tmax);
                    match maybe_hit {
                        Some(hit) => {
                            let frame = TangentFrame::from_normal(hit.normal);
                            // let wi = (primary.ray.origin - hit.point).normalized();
                            let wi = -primary.ray.direction;
                            let local_wi = frame.to_local(&wi).normalized();
                            let emission = self.world.get_material(hit.material).emission(
                                primary.lambda,
                                hit.uv,
                                primary.transport_mode,
                                local_wi,
                                None,
                            );
                            *isect = IntersectionData::Surface {
                                lambda: primary.lambda,
                                local_wi,
                                point: hit.point,
                                normal: hit.normal,
                                instance_id: hit.instance_id,
                                emission,
                                throughput: mult * primary.throughput,
                                transport_mode: primary.transport_mode,
                                local_wo: None,
                                material: hit.material,
                                uv: hit.uv,
                                camera_pixel_id: primary.cam_and_pixel_id,
                            }
                        }
                        None => {
                            // handle env hit or no hit, depending on it the ray limit was infinity or not
                            if primary.ray.tmax.is_infinite() {
                                *isect = IntersectionData::Environment {
                                    uv: direction_to_uv(primary.ray.direction),
                                    lambda: primary.lambda,
                                }
                            } else {
                                *isect = IntersectionData::Empty;
                            }
                        }
                    }
                }
            });
    }

    pub fn nee_pass(
        &self,
        light_samples: usize,
        intersection_buffer: &IntersectionBuffer,
        shadow_buffer: &mut ShadowRayBuffer,
    ) {
        // pick a single light to try and sample?
        let (light, _light_pick_pdf) = self
            .world
            .pick_random_light(Sample1D::new_random_sample())
            .unwrap();
        // queues up NEE shadow rays
        shadow_buffer
            .rays
            .par_iter_mut()
            .zip(intersection_buffer.intersections.par_iter())
            .for_each(|(maybe_shadow, isect)| {
                if let IntersectionData::Surface { lambda, point, .. } = isect {
                    let (wo, psa_pdf) = light.sample(Sample2D::new_random_sample(), *point);
                    *maybe_shadow = Some((
                        PrimaryRay {
                            ray: Ray::new(*point, wo),
                            throughput: psa_pdf.0,
                            cam_and_pixel_id: None,
                            buffer_idx: 0,
                            transport_mode: TransportMode::Importance,
                            lambda: *lambda,
                        },
                        VisibilityTestStatus::Unchecked,
                        None,
                    ));
                } else {
                    *maybe_shadow = None;
                }
            });
    }
    pub fn visibility_intersection_pass(&self, buffer: &mut ShadowRayBuffer) {
        // actually performs intersection with scene, putting results in the intersection buffer
        buffer.rays.par_iter_mut().for_each(|shadow| {
            if shadow.is_none() {
                return;
            }
            let local_ray = shadow.unwrap();
            let primary = local_ray.0;
            let ray = primary.ray;
            let maybe_hit = self.world.hit(ray, 0.0001, ray.tmax);
            match maybe_hit {
                Some(hit) => {
                    let frame = TangentFrame::from_normal(hit.normal);
                    // let wi = (primary.ray.origin - hit.point).normalized();
                    let wi = -ray.direction;
                    let local_wi = frame.to_local(&wi).normalized();
                    let emission = self.world.get_material(hit.material).emission(
                        primary.lambda,
                        hit.uv,
                        primary.transport_mode,
                        local_wi,
                        None,
                    );
                    let status = if emission.0 > 0.0 {
                        VisibilityTestStatus::Visible
                    } else {
                        VisibilityTestStatus::Blocked
                    };
                    let local_isect = IntersectionData::Surface {
                        lambda: primary.lambda,
                        local_wi,
                        point: hit.point,
                        normal: hit.normal,
                        instance_id: hit.instance_id,
                        emission,
                        throughput: primary.throughput,
                        transport_mode: primary.transport_mode,
                        local_wo: None,
                        material: hit.material,
                        uv: hit.uv,
                        camera_pixel_id: None,
                    };
                    *shadow = Some((
                        primary,
                        status,
                        Some(ConnectionData::Intersection(local_isect)),
                    ));
                }
                None => {
                    // handle no hit case
                    *shadow = Some((primary, VisibilityTestStatus::Visible, None));
                }
            }
        });
    }
    pub fn shading_pass(
        &self,
        intersection_buffer: &IntersectionBuffer,
        shadow_buffer: &ShadowRayBuffer,
        _cam_bounds: Bounds2D,
        buffer: &mut ShadingResultBuffer,
        sample_buffer: &mut SampleBounceBuffer,
        max_bounces: usize,
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
            .for_each(|(index, (shading_result, isect_data))| match isect_data {
                IntersectionData::Surface {
                    local_wi,
                    uv,
                    normal,
                    lambda,
                    emission,
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
                    let frame = TangentFrame::from_normal(*normal);
                    let nee_data = shadow_buffer.rays[index];
                    // estimate throughput from this path
                    let mut contribution = SingleEnergy::ZERO;
                    if let Some((shadow_primary, status, data)) = nee_data {
                        match status {
                            VisibilityTestStatus::Visible => {
                                if let Some(ConnectionData::Intersection(
                                    IntersectionData::Surface {
                                        emission,
                                        local_wi: light_local_wi,
                                        ..
                                    },
                                )) = data
                                {
                                    // since we assigned the throughput as the pdf earlier, when doing the NEE intersection pass
                                    let light_psa_pdf = shadow_primary.throughput;
                                    let wo = -light_local_wi;
                                    let f =
                                        material.f(*lambda, *uv, *transport_mode, *local_wi, wo);
                                    let pdf = material.scatter_pdf(
                                        *lambda,
                                        *uv,
                                        *transport_mode,
                                        *local_wi,
                                        wo,
                                    );
                                    // do MIS here
                                    let cos_i = wo.z().abs();
                                    let weight = power_heuristic(light_psa_pdf, pdf.0);
                                    contribution = cos_i * emission * weight * f / light_psa_pdf;
                                } else {
                                }
                            }
                            _ => {}
                        }
                    }

                    // calculate data for next bounce
                    let f = local_wo.map_or(SingleEnergy::ZERO, |v| {
                        material.f(*lambda, *uv, *transport_mode, *local_wi, v)
                    });
                    let pdf = local_wo.map_or(0.0.into(), |v| {
                        material.scatter_pdf(*lambda, *uv, *transport_mode, *local_wi, v)
                    });
                    let cos_i = local_wo.map_or(0.0, |wo| wo.z().abs());

                    *shading_result = (
                        *throughput * (*emission + contribution),
                        *throughput * cos_i * f / pdf.0,
                        local_wo.map(|v| frame.to_world(&v).normalized()),
                    );
                }
                IntersectionData::Environment { uv, lambda, .. } => {
                    let energy = self.world.environment.emission(*uv, *lambda);

                    *shading_result = (energy, SingleEnergy::ZERO, None);
                }
                _ => {
                    *shading_result = (SingleEnergy::ZERO, SingleEnergy::ZERO, None);
                }
            });

        // generate new rays based on if the interaction was one that bounced, based on the isect data
        // cannot be parallelized because we're also modifying the film at the same time
        for (i, ray) in rays.rays.iter_mut().enumerate() {
            if ray.is_none() {
                continue;
            }
            let uray = ray.unwrap();
            let isect = intersection_buffer.intersections[i];
            let (light, next_throughput, wo) = buffer.data[i];
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
            let cam_and_pixel_id = uray.cam_and_pixel_id;
            if let Some((cam_id, pixel)) = cam_and_pixel_id {
                let lambda = uray.lambda;
                let idx = match pixel {
                    Pixel::Id(id) => id,
                    Pixel::UV(u, v) => {
                        (u * film.width as f32) as usize
                            + ((v * film.height as f32) as usize * film.width)
                    }
                };
                if let Some(v) = film.buffer.get_mut(idx) {
                    *v += XYZColor::from(SingleWavelength::new(lambda, light));
                }
            }

            if let IntersectionData::Surface { point, lambda, .. } = isect {
                if wo.is_none() || sample_buffer.sample_count[uray.buffer_idx].1 >= max_bounces {
                    *ray = None;
                    sample_buffer.sample_count[uray.buffer_idx].1 = 0; // reset bounce count
                    sample_buffer.sample_count[uray.buffer_idx].0 += 1; // increment sample count
                    continue;
                }

                *ray = Some(PrimaryRay {
                    ray: Ray::new(point, wo.unwrap()),
                    buffer_idx: uray.buffer_idx,
                    lambda,
                    throughput: next_throughput.0,
                    transport_mode: uray.transport_mode,
                    cam_and_pixel_id,
                });
                sample_buffer.sample_count[uray.buffer_idx].1 += 1; // increment bounce count
            } else {
                sample_buffer.sample_count[uray.buffer_idx].1 = 0; // reset bounce count
                sample_buffer.sample_count[uray.buffer_idx].0 += 1; // increment sample count
                *ray = None;
            }
        }
    }
    pub fn finalize_pass(
        &self,
        film: &mut Film<XYZColor>,
        sample_buffer: &SampleBounceBuffer,
        topleft: (usize, usize),
    ) {
        for (idx, (sample, _)) in sample_buffer.sample_count.iter().enumerate() {
            let (sample_x, sample_y) = (idx % KERNEL_WIDTH, idx / KERNEL_WIDTH);
            let (film_x, film_y) = (topleft.0 + sample_x, topleft.1 + sample_y);
            film.buffer[film_y * film.width + film_x] /= *sample as f32;
        }
    }
}
