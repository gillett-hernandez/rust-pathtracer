
use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::*;
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};

use std::sync::Arc;

fn evaluate_direct_importance(
    world: &Arc<World>,
    camera_pick: Sample1D,
    lens_sample: Sample2D,
    lambda: f32,
    beta: f32,
    material: &MaterialEnum,
    wi: Vec3,
    hit: &HitRecord,
    frame: &TangentFrame,
    samples: &mut Vec<(Sample, CameraId)>,
    profile: &mut Profile,
) {
    let (camera, camera_id, camera_pick_pdf) = world
        .pick_random_camera(camera_pick)
        .expect("camera pick failed");
    if let Some(camera_surface) = camera.get_surface() {
        let (point_on_lens, lens_normal, pdf) = camera_surface.sample_surface(lens_sample);
        // maybe resample to a better direction, if we had a outer->inner lens acceleration structure
        let camera_pdf = pdf * *camera_pick_pdf;
        if *camera_pdf == 0.0 {
            // go to next pick
            return;
        }
        let direction = (point_on_lens - hit.point).normalized();

        // camera_surface.material_id
        let wo_to_camera = frame.to_local(&direction);

        let (reflectance, scatter_pdf_into_camera) =
            material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, wo_to_camera);

        // trace!("picked valid camera, {:?}, {:?}", direction, pdf);
        // generate point on camera, then see if it can be connected to.
        // println!("hit {:?}", &hit);
        profile.shadow_rays += 1;
        if veach_v(world, point_on_lens, hit.point) {
            let weight = power_heuristic(*camera_pdf, *scatter_pdf_into_camera);

            // correctly connected.
            let candidate_camera_ray = Ray::new(point_on_lens, -direction);
            let pixel_uv = camera.get_pixel_for_ray(candidate_camera_ray, lambda);
            let (we, _pdf) = camera.eval_we(lambda, lens_normal, point_on_lens, hit.point);
            // trace!(
            //     " weight {}, uv for ray {:?} is {:?}",
            //     weight,
            //     candidate_camera_ray,
            //     pixel_uv
            // );
            if let Some(uv) = pixel_uv {
                debug_assert!(
                    !camera_pdf.is_nan() && !weight.is_nan(),
                    "{:?}, {}",
                    camera_pdf,
                    weight
                );
                let energy =
                    reflectance * beta * wo_to_camera.z().abs() * we * weight / *camera_pdf;
                debug_assert!(energy.is_finite());
                let sample = XYZColor::from(SingleWavelength::new(lambda, energy));

                // println!("adding camera sample to splatting list");
                samples.push((Sample::LightSample(sample, uv), camera_id as CameraId));
            }
        }
    }
}

pub struct LightTracingIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub camera_samples: u16,
    pub wavelength_bounds: Bounds1D,
}

impl GenericIntegrator for LightTracingIntegrator {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        _settings: &RenderSettings,
        _camera_sample: ((f32, f32), CameraId),
        _sample_id: usize,
        samples: &mut Vec<(Sample, CameraId)>,
        profile: &mut Profile,
    ) -> XYZColor {
        // setup: decide light, decide wavelength, emit ray from light, connect light ray vertices to camera.
        let wavelength_sample = sampler.draw_1d();
        let light_pick_sample = sampler.draw_1d();

        let env_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;
        let light_g_term: f32;

        let (light_pick_sample, sample_world) =
            light_pick_sample.choose(env_sampling_probability, true, false);
        let light_type;
        if !sample_world {
            let (light, pick_pdf) = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal, area_pdf) =
                light.sample_surface(sampler.draw_2d());

            let mat_id = light.get_material_id();
            let material = self.world.get_material(mat_id);
            // println!("sampled light emission in instance light branch");
            let tmp_sampled = material
                .sample_emission(
                    light_surface_point,
                    light_surface_normal,
                    self.wavelength_bounds,
                    sampler.draw_2d(),
                    wavelength_sample,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "emission sample failed, light is {:?} material is {:?}",
                        light, mat_id
                    )
                });
            light_g_term = (light_surface_normal * (tmp_sampled.0).direction).abs();
            sampled = (
                tmp_sampled.0,
                tmp_sampled.1,
                tmp_sampled.2 * *pick_pdf * *area_pdf, // should be a throughput pdf i think. since it's projected solid angle * area
                tmp_sampled.3,
            );
            light_type = LightSourceType::Instance;
        } else {
            // sample world env
            // println!("sampled light emission in world light branch");
            // println!("sampling world, world radius is {}", world_radius);
            let world_radius = self.world.radius;
            let world_center = self.world.center;
            sampled = self.world.environment.sample_emission(
                world_radius,
                world_center,
                sampler.draw_2d(),
                sampler.draw_2d(),
                self.wavelength_bounds,
                wavelength_sample,
            );
            light_g_term = 1.0;
            light_type = LightSourceType::Environment;
            // sampled = (tmp_sampled.0, tmp_sampled.1, tmp_sampled.2);
        };
        profile.light_rays += 1;
        let light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;
        if radiance == 0.0 {
            return XYZColor::BLACK;
        }
        let light_pdf = sampled.2;
        let lambda_pdf = sampled.3;

        // light loop here
        let mut path: Vec<SurfaceVertex<f32, f32>> =
            Vec::with_capacity(1 + self.max_bounces as usize);

        path.push(SurfaceVertex::new(
            VertexType::LightSource(light_type),
            light_ray.time,
            lambda,
            Vec3::ZERO,
            light_ray.origin,
            light_ray.direction,
            UV(0.0, 0.0),
            MaterialId::Light(0),
            0,
            radiance / *lambda_pdf,
            light_pdf,
            PDF::new(0.0),
            light_g_term,
            0,
            0,
        ));
        random_walk(
            light_ray,
            lambda,
            self.max_bounces,
            radiance / *light_pdf / *lambda_pdf,
            // radiance ,
            TransportMode::Radiance,
            sampler,
            &self.world,
            &mut path,
            0,
            profile,
            true,
        );

        if let Some(SurfaceVertex {
            vertex_type: VertexType::Camera,
            ..
        }) = path.get(1)
        {
            trace!("{:?}\n\n", path);
        }

        // let mut sum = ;
        // let mut multiplier = 1.0;

        for (index, vertex) in path.iter().enumerate() {
            if index == 0 {
                continue;
            }
            let prev_vertex = path[index - 1];
            let beta = vertex.throughput;

            // for every vertex past the 1st one (which is on the light), evaluate the direct importance at that vertex
            match vertex.vertex_type {
                VertexType::Light => {
                    // generic vertex along light path, handle normally
                    let hit = HitRecord::from(*vertex);
                    let frame = TangentFrame::from_normal(hit.normal);

                    let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                    let wi = frame.to_local(&dir_to_prev);
                    let material = self.world.get_material(vertex.material_id);

                    for _ in 0..self.camera_samples {
                        evaluate_direct_importance(
                            &self.world,
                            sampler.draw_1d(),
                            sampler.draw_2d(),
                            vertex.lambda,
                            beta,
                            material,
                            wi,
                            &hit,
                            &frame,
                            samples,
                            profile,
                        );
                    }
                    // if self.camera_samples > 0 {
                    //     trace!("added {} camera samples", samples.len());
                    // }
                }
                VertexType::Camera => {
                    assert!(matches!(vertex.material_id, MaterialId::Camera(_)));
                    let camera_id: usize = vertex.material_id.into();
                    // directly hit camera
                    // maybe resample to a better direction, if we had a outer->inner lens acceleration structure
                    match prev_vertex.vertex_type {
                        VertexType::LightSource(_light_source_type) => {
                            // camera to light source direct connection.

                            // if light_source_type == LightSourceType::Environment {
                            //     // do resampling maybe?
                            // }
                            // TODO: figure out some way to make lens_normal work for realistic cameras.
                            // currently it's based on the disk instance in the scene.
                            let camera = self.world.get_camera(camera_id);
                            // this could also be a point on the env.
                            let point_on_light = prev_vertex.point;

                            let (point_on_lens, lens_normal) = (vertex.point, vertex.normal);

                            let direction = (point_on_lens - point_on_light).normalized();

                            let hypothetical_camera_ray = Ray::new(point_on_lens, -direction);
                            let pixel_uv =
                                camera.get_pixel_for_ray(hypothetical_camera_ray, lambda);
                            let (we, _pdf) =
                                camera.eval_we(lambda, lens_normal, point_on_lens, point_on_light);
                            trace!(
                                "uv for ray {:?} is {:?}, we = {}",
                                hypothetical_camera_ray,
                                pixel_uv,
                                we
                            );
                            if let Some(uv) = pixel_uv {
                                let energy = beta * we;
                                debug_assert!(energy.is_finite());
                                let sample = XYZColor::from(SingleWavelength::new(lambda, energy));

                                // println!("adding camera sample to splatting list");
                                samples
                                    .push((Sample::LightSample(sample, uv), camera_id as CameraId));
                            }
                        }
                        VertexType::Light => {
                            // hit camera from vertex in scene, analogous to hitting a light while random walking in PT
                            // TODO
                        }
                        _ => unreachable!(),
                    }
                }
                VertexType::LightSource(_) => {
                    lazy_static! {
                        static ref LOGGED_CELL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    }
                    if !LOGGED_CELL.fetch_or(true, std::sync::atomic::Ordering::AcqRel) {
                        warn!("hit light source while doing light tracing");
                    }
                    // could potentially add energy to the path if light sources are hit while tracing
                }
                VertexType::Eye => unreachable!(),
            }
        }
        XYZColor::BLACK
    }
}
