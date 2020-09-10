use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::*;
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::TransportMode;
use crate::{INTERSECTION_TIME_OFFSET, NORMAL_OFFSET};

use std::f32::INFINITY;
use std::sync::Arc;

fn evaluate_direct_importance(
    world: &Arc<World>,
    camera_pick: Sample1D,
    lens_sample: Sample2D,
    lambda: f32,
    beta: SingleEnergy,
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
        let (point_on_lens, _lens_normal, pdf) = camera_surface.sample_surface(lens_sample);
        let camera_pdf = pdf * camera_pick_pdf;
        if camera_pdf.0 == 0.0 {
            // go to next pick
            return;
        }
        let direction = (point_on_lens - hit.point).normalized();

        // this should be the same as the other method, but maybe not.
        // camera_surface.material_id
        let camera_wo = frame.to_local(&direction);
        let reflectance = material.f(&hit, wi, camera_wo);
        let dropoff = camera_wo.z().max(0.0);
        if dropoff == 0.0 {
            return;
        }
        // println!("picked valid camera, {:?}, {:?}", direction, pdf);
        // generate point on camera, then see if it can be connected to.
        // println!("hit {:?}", &hit);
        profile.shadow_rays += 1;
        if veach_v(&world, point_on_lens, hit.point) {
            let scatter_pdf_into_camera = material.scatter_pdf(&hit, wi, camera_wo);
            let weight = power_heuristic(camera_pdf.0, scatter_pdf_into_camera.0);

            // correctly connected.
            let candidate_camera_ray = Ray::new(point_on_lens, -direction);
            let pixel_uv = camera.get_pixel_for_ray(candidate_camera_ray);
            // println!(
            //     " weight {}, uv for ray {:?} is {:?}",
            //     weight, candidate_camera_ray, pixel_uv
            // );
            if let Some(uv) = pixel_uv {
                debug_assert!(
                    !camera_pdf.is_nan() && !weight.is_nan(),
                    "{:?}, {}",
                    camera_pdf,
                    weight
                );
                let energy = reflectance * beta * dropoff * weight / camera_pdf.0;
                debug_assert!(energy.0.is_finite());
                let sw = SingleWavelength::new(lambda, energy);
                let ret = (Sample::LightSample(sw, uv), camera_id as CameraId);
                // println!("adding camera sample to splatting list");
                samples.push(ret);
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
        mut samples: &mut Vec<(Sample, CameraId)>,
        mut profile: &mut Profile,
    ) -> SingleWavelength {
        // setup: decide light, decide wavelength, emit ray from light, connect light ray vertices to camera.
        let wavelength_sample = sampler.draw_1d();
        let light_pick_sample = sampler.draw_1d();

        let env_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;
        let light_g_term: f32;

        let (light_pick_sample, sample_world) =
            light_pick_sample.choose(env_sampling_probability, true, false);
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
                    VISIBLE_RANGE,
                    sampler.draw_2d(),
                    wavelength_sample,
                )
                .expect(&format!(
                    "emission sample failed, light is {:?} material is {:?}",
                    light, mat_id
                ));
            light_g_term = (light_surface_normal * (&tmp_sampled.0).direction).abs();
            sampled = (
                tmp_sampled.0,
                tmp_sampled.1,
                tmp_sampled.2 * pick_pdf * area_pdf,
                tmp_sampled.3,
            );
        } else {
            // sample world env
            // println!("sampled light emission in world light branch");
            // println!("sampling world, world radius is {}", world_radius);
            let world_radius = self.world.get_world_radius();
            let world_center = self.world.get_center();
            sampled = self.world.environment.sample_emission(
                world_radius,
                world_center,
                sampler.draw_2d(),
                sampler.draw_2d(),
                VISIBLE_RANGE,
                wavelength_sample,
            );
            light_g_term = 1.0;
            // sampled = (tmp_sampled.0, tmp_sampled.1, tmp_sampled.2);
        };
        profile.light_rays += 1;
        let light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;
        if radiance.0 == 0.0 {
            return SingleWavelength::BLACK;
        }
        let light_pdf = sampled.2;
        let lambda_pdf = sampled.3;

        // light loop here
        let mut path: Vec<Vertex> = Vec::with_capacity(1 + self.max_bounces as usize);

        path.push(Vertex::new(
            VertexType::Camera,
            light_ray.time,
            lambda,
            Vec3::ZERO,
            light_ray.origin,
            light_ray.direction,
            (0.0, 0.0),
            MaterialId::Camera(0),
            0,
            SingleEnergy::ONE,
            light_pdf.0,
            0.0,
            light_g_term,
        ));
        let _ = random_walk(
            light_ray,
            lambda,
            self.max_bounces,
            radiance * light_pdf.0 / lambda_pdf.0,
            TransportMode::Radiance,
            sampler,
            &self.world,
            &mut path,
            0,
            &mut profile,
        );

        // let mut sum = ;
        let mut multiplier = 1.0;

        for (index, vertex) in path.iter().enumerate() {
            if index == 0 {
                continue;
            }
            let prev_vertex = path[index - 1];
            if index == 1 {
                multiplier =
                    vertex.local_wi.z() / (vertex.point - prev_vertex.point).norm_squared();
            }
            let beta = vertex.throughput * multiplier;

            // for every vertex past the 1st one (which is on the camera), evaluate the direct illumination at that vertex, and if it hits a light evaluate the added energy
            if let VertexType::LightSource(light_source) = vertex.vertex_type {
                if light_source == LightSourceType::Environment {
                    // let wo = -vertex.local_wi;
                    // let uv = direction_to_uv(wo);
                    // let emission = self.world.environment.emission(uv, lambda);
                    // sum.energy += emission * beta;
                } else {
                    // let hit = HitRecord::from(*vertex);
                    // let frame = TangentFrame::from_normal(hit.normal);
                    // let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                    // let maybe_dir_to_next = path
                    //     .get(index + 1)
                    //     .map(|v| (v.point - vertex.point).normalized());
                    // let wi = frame.to_local(&dir_to_prev);
                    // let wo = maybe_dir_to_next.map(|dir| frame.to_local(&dir));
                    // let material = self.world.get_material(vertex.material_id);

                    // let emission = material.emission(&hit, wi, wo);

                    // if emission.0 > 0.0 {
                    //     // this will likely never get triggered, since hitting a light source is handled in the above branch
                    //     if prev_vertex.pdf_forward <= 0.0 || self.camera_samples == 0 {
                    //         sum.energy += beta * emission;
                    //         debug_assert!(!sum.energy.is_nan());
                    //     } else {
                    //         let hit_primitive = self.world.get_primitive(hit.instance_id);
                    //         // // println!("{:?}", hit);
                    //         let pdf = hit_primitive.psa_pdf(
                    //             prev_vertex.normal,
                    //             prev_vertex.point,
                    //             hit.point,
                    //         );
                    //         let weight = power_heuristic(prev_vertex.pdf_forward, pdf.0);
                    //         debug_assert!(
                    //             !pdf.is_nan() && !weight.is_nan(),
                    //             "{:?}, {}",
                    //             pdf,
                    //             weight
                    //         );
                    //         sum.energy += beta * emission * weight;
                    //         debug_assert!(!sum.energy.is_nan());
                    //     }
                    // }
                }
            } else {
                let hit = HitRecord::from(*vertex);
                let frame = TangentFrame::from_normal(hit.normal);
                let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                let maybe_dir_to_next = path
                    .get(index + 1)
                    .map(|v| (v.point - vertex.point).normalized());
                let wi = frame.to_local(&dir_to_prev);
                let wo = maybe_dir_to_next.map(|dir| frame.to_local(&dir));
                let material = self.world.get_material(vertex.material_id);

                // let emission = material.emission(&hit, wi, wo);

                // if emission.0 > 0.0 {
                //     // this will likely never get triggered, since hitting a light source is handled in the above branch
                //     if prev_vertex.pdf_forward <= 0.0 || self.camera_samples == 0 {
                //         sum.energy += vertex.throughput * emission;
                //         debug_assert!(!sum.energy.is_nan());
                //     } else {
                //         let hit_primitive = self.world.get_primitive(hit.instance_id);
                //         // // println!("{:?}", hit);
                //         let pdf =
                //             hit_primitive.psa_pdf(prev_vertex.normal, prev_vertex.point, hit.point);
                //         let weight = power_heuristic(prev_vertex.pdf_forward, pdf.0);
                //         debug_assert!(!pdf.is_nan() && !weight.is_nan(), "{:?}, {}", pdf, weight);
                //         sum.energy += vertex.throughput * emission * weight;
                //         debug_assert!(!sum.energy.is_nan());
                //     }
                // }

                if self.camera_samples > 0 {
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
                            &mut samples,
                            &mut profile,
                        );
                    }
                }
            }
        }
        SingleWavelength::BLACK
    }
}
