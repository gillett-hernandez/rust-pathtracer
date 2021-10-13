use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::{random_walk, veach_v, LightSourceType, SurfaceVertex, VertexType};
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};
use crate::math::*;
use crate::world::TransportMode;
// use crate::world::EnvironmentMap;

use std::f32::INFINITY;
use std::sync::Arc;

pub struct PathTracingIntegrator {
    pub min_bounces: u16,
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub light_samples: u16,
    pub only_direct: bool,
    pub wavelength_bounds: Bounds1D,
}

impl PathTracingIntegrator {
    fn estimate_direct_illumination(
        &self,
        hit: &HitRecord,
        frame: &TangentFrame,
        wi: Vec3,
        material: &MaterialEnum,
        throughput: SingleEnergy,
        light_pick_sample: Sample1D,
        additional_light_sample: Sample2D,
        profile: &mut Profile,
    ) -> SingleEnergy {
        if let Some((light, light_pick_pdf)) = self.world.pick_random_light(light_pick_sample) {
            // determine pick pdf
            // as of now the pick pdf is just num lights, however if it were to change this would be where it should change.
            // sample the primitive from hit_point
            // let (direction, light_pdf) = light.sample(sampler.draw_2d(), hit.point);
            let (point_on_light, normal, light_area_pdf) =
                light.sample_surface(additional_light_sample);
            debug_assert!(light_area_pdf.0.is_finite());
            if light_area_pdf.0 == 0.0 {
                return SingleEnergy::ZERO;
            }
            // direction is from shading point to light
            let direction = (point_on_light - hit.point).normalized();
            // direction is already in world space.
            // direction is also oriented away from the shading point already, so no need to negate directions until later.
            let local_light_direction = frame.to_local(&direction);
            let light_vertex_wi = TangentFrame::from_normal(normal).to_local(&(-direction));

            let dropoff = light_vertex_wi.z().abs();
            if dropoff == 0.0 {
                return SingleEnergy::ZERO;
            }
            // since direction is already in world space, no need to call frame.to_world(direction) in the above line
            let (reflectance, scatter_pdf_for_light_ray) = material.bsdf(
                hit.lambda,
                hit.uv,
                hit.transport_mode,
                wi,
                local_light_direction,
            );
            // if reflectance.0 < 0.00001 {
            //     // if reflectance is 0 for all components, skip this light sample
            //     continue;
            // }

            let pdf = light.psa_pdf(
                hit.normal * (point_on_light - hit.point).normalized(),
                hit.point,
                point_on_light,
            );
            let light_pdf = pdf * light_pick_pdf; // / light_vertex_wi.z().abs();
            if light_pdf.0 == 0.0 {
                // println!("light pdf was 0");
                // go to next pick
                return SingleEnergy::ZERO;
            }

            let light_material = self.world.get_material(light.get_material_id());
            let emission =
                light_material.emission(hit.lambda, hit.uv, hit.transport_mode, light_vertex_wi);
            // this should be the same as the other method, but maybe not.
            if emission.0 == 0.0 {
                return SingleEnergy::ZERO;
            }

            profile.shadow_rays += 1;
            if veach_v(&self.world, point_on_light, hit.point) {
                let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);

                debug_assert!(emission.0 >= 0.0);
                // successful_light_samples += 1;
                let v = reflectance * throughput * dropoff * emission * weight / light_pdf.0;
                debug_assert!(
                    v.0.is_finite(),
                    "{:?},{:?},{:?},{:?},{:?},{:?},",
                    reflectance,
                    throughput,
                    dropoff,
                    emission,
                    weight,
                    light_pdf.0
                );
                return v;
                // debug_assert!(
                //     !light_contribution.0.is_nan(),
                //     "l {:?} r {:?} b {:?} d {:?} s {:?} w {:?} p {:?} ",
                //     light_contribution,
                //     reflectance,
                //     beta,
                //     dropoff,
                //     emission,
                //     weight,
                //     light_pdf
                // );
            }
        }
        SingleEnergy::ZERO
    }

    fn estimate_direct_illumination_from_world(
        &self,
        lambda: f32,
        hit: &HitRecord,
        frame: &TangentFrame,
        wi: Vec3,
        material: &MaterialEnum,
        throughput: SingleEnergy,
        sample: Sample2D,
        profile: &mut Profile,
    ) -> SingleEnergy {
        let (uv, light_pdf) = self
            .world
            .environment
            .sample_env_uv_given_wavelength(sample, lambda);
        // direction is the direction to the sampled point on the environment
        let direction = uv_to_direction(uv);
        let local_wo = frame.to_local(&direction);

        let local_dropoff = local_wo.z();

        let (reflectance, scatter_pdf_for_light_ray) =
            material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, local_wo);

        profile.shadow_rays += 1;
        // TODO: add support for passthrough material, such that it doesn't fully interfere with direct illumination
        if let Some(mut _light_hit) =
            self.world
                .hit(Ray::new(hit.point, direction), 0.00001, INFINITY)
        {
            return 0.0.into();
        /*
        light_hit.lambda = lambda;
        // handle case where we intended to hit the world but instead hit a light
        let material = self.world.get_material(light_hit.material);

        let point_on_light = light_hit.point;
        let light_frame = TangentFrame::from_normal(light_hit.normal);
        let light_wi = light_frame.to_local(&-direction);
        let dropoff = light_wi.z().abs();
        if dropoff == 0.0 {
            return SingleEnergy::ZERO;
        }
        // if reflectance.0 < 0.00001 {
        //     // if reflectance is 0 for all components, skip this light sample
        //     continue;
        // }
        let emission = material.emission(
            light_hit.lambda,
            light_hit.uv,
            light_hit.transport_mode,
            light_wi,
        );
        if emission.0 > 0.0 {
            let light = self.world.get_primitive(light_hit.instance_id);
            let pdf = light.psa_pdf(
                hit.normal * (point_on_light - hit.point).normalized(),
                hit.point,
                point_on_light,
            );

            if pdf.0 == 0.0 {
                return SingleEnergy::ZERO;
            }
            let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);
            reflectance * throughput * dropoff * emission * weight / light_pdf.0
        } else {
            SingleEnergy::ZERO
        } */
        } else {
            // successfully hit nothing, which is to say, hit the world
            let emission = self.world.environment.emission(uv, lambda);

            let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);
            let v =
                reflectance * local_dropoff.abs() * throughput * emission * weight / light_pdf.0;
            debug_assert!(
                v.0.is_finite(),
                "{:?},{:?},{:?},{:?},{:?},{:?},",
                reflectance,
                local_dropoff,
                throughput,
                emission,
                weight,
                light_pdf.0
            );
            v
        }
    }

    pub fn estimate_direct_illumination_with_loop(
        &self,
        lambda: f32,
        hit: &HitRecord,
        frame: &TangentFrame,
        wi: Vec3,
        material: &MaterialEnum,
        throughput: SingleEnergy,
        sampler: &mut Box<dyn Sampler>,
        mut profile: &mut Profile,
    ) -> SingleEnergy {
        let mut light_contribution = SingleEnergy::ZERO;
        let env_sampling_probability = self.world.get_env_sampling_probability();
        if self.world.lights.len() == 0 && env_sampling_probability == 0.0 {
            return SingleEnergy::ZERO;
        }
        for _i in 0..self.light_samples {
            let (light_pick_sample, sample_world) =
                sampler
                    .draw_1d()
                    .choose(env_sampling_probability, true, false);
            // decide whether to sample the lights or the world
            if sample_world {
                // light_contribution += self.world.environment.sample
                light_contribution += self.estimate_direct_illumination_from_world(
                    lambda,
                    hit,
                    frame,
                    wi,
                    material,
                    throughput,
                    sampler.draw_2d(),
                    &mut profile,
                );
            } else {
                light_contribution += self.estimate_direct_illumination(
                    &hit,
                    &frame,
                    wi,
                    material,
                    throughput,
                    light_pick_sample,
                    sampler.draw_2d(),
                    &mut profile,
                );
            }
            debug_assert!(
                light_contribution.0.is_finite(),
                "{:?}, {}, {:?}, {:?}, {:?}",
                light_contribution,
                sample_world,
                hit.material,
                material.get_name(),
                wi,
            );
        }
        light_contribution
    }
}

impl SamplerIntegrator for PathTracingIntegrator {
    fn color(
        &self,
        mut sampler: &mut Box<dyn Sampler>,
        camera_sample: ((f32, f32), CameraId),
        _sample_id: usize,
        mut profile: &mut Profile,
    ) -> XYZColor {
        profile.camera_rays += 1;

        let mut sum = SingleWavelength::new_from_range(sampler.draw_1d().x, self.wavelength_bounds);
        let lambda = sum.lambda;

        let camera_id = camera_sample.1;
        let camera = self.world.get_camera(camera_id as usize);
        let film_sample = Sample2D::new(
            (camera_sample.0).0.clamp(0.0, 1.0 - std::f32::EPSILON),
            (camera_sample.0).1.clamp(0.0, 1.0 - std::f32::EPSILON),
        );

        let (camera_ray, _lens_normal, throughput_and_pdf) =
            camera.sample_we(film_sample, &mut sampler, sum.lambda);
        let camera_pdf = throughput_and_pdf;
        if camera_pdf.0 == 0.0 {
            return XYZColor::BLACK;
        }

        let mut path: Vec<SurfaceVertex> = Vec::with_capacity(1 + self.max_bounces as usize);

        path.push(SurfaceVertex::new(
            VertexType::Camera,
            camera_ray.time,
            lambda,
            Vec3::ZERO,
            camera_ray.origin,
            camera_ray.direction,
            (0.0, 0.0),
            MaterialId::Camera(0),
            0,
            SingleEnergy::from(throughput_and_pdf.0),
            0.0,
            0.0,
            1.0,
        ));
        let _ = random_walk(
            camera_ray,
            lambda,
            self.max_bounces,
            SingleEnergy::from(throughput_and_pdf.0),
            TransportMode::Importance,
            sampler,
            &self.world,
            &mut path,
            self.min_bounces,
            &mut profile,
        );

        for (index, vertex) in path.iter().enumerate() {
            if index == 0 {
                continue;
            }
            let prev_vertex = path[index - 1];
            // for every vertex past the 1st one (which is on the camera), evaluate the direct illumination at that vertex, and if it hits a light evaluate the added energy
            if let VertexType::LightSource(light_source) = vertex.vertex_type {
                if light_source == LightSourceType::Environment {
                    let wo = vertex.local_wi;
                    let uv = direction_to_uv(wo);
                    let emission = self.world.environment.emission(uv, lambda);
                    sum.energy += emission * vertex.throughput;
                } else {
                    let hit = HitRecord::from(*vertex);
                    let frame = TangentFrame::from_normal(hit.normal);
                    let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                    let _maybe_dir_to_next = path
                        .get(index + 1)
                        .map(|v| (v.point - vertex.point).normalized());
                    let wi = frame.to_local(&dir_to_prev);
                    let material = self.world.get_material(vertex.material_id);

                    let emission = material.emission(hit.lambda, hit.uv, hit.transport_mode, wi);

                    if emission.0 > 0.0 {
                        if prev_vertex.pdf_forward <= 0.0 || self.light_samples == 0 {
                            sum.energy += vertex.throughput * emission;
                            debug_assert!(!sum.energy.is_nan());
                        } else {
                            let hit_primitive = self.world.get_primitive(hit.instance_id);
                            // // println!("{:?}", hit);
                            let pdf = hit_primitive.psa_pdf(
                                prev_vertex.normal * (hit.point - prev_vertex.point).normalized(),
                                prev_vertex.point,
                                hit.point,
                            );
                            let weight = power_heuristic(prev_vertex.pdf_forward, pdf.0);
                            debug_assert!(
                                !pdf.is_nan() && !weight.is_nan(),
                                "{:?}, {}",
                                pdf,
                                weight
                            );
                            sum.energy += vertex.throughput * emission * weight;
                            debug_assert!(!sum.energy.is_nan());
                        }
                    }
                }
            } else {
                let hit = HitRecord::from(*vertex);
                let frame = TangentFrame::from_normal(hit.normal);
                let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                let _maybe_dir_to_next = path
                    .get(index + 1)
                    .map(|v| (v.point - vertex.point).normalized());
                let wi = frame.to_local(&dir_to_prev);
                let material = self.world.get_material(vertex.material_id);

                let emission = material.emission(hit.lambda, hit.uv, hit.transport_mode, wi);

                if emission.0 > 0.0 {
                    // this will likely never get triggered, since hitting a light source is handled in the above branch
                    if prev_vertex.pdf_forward <= 0.0 || self.light_samples == 0 {
                        sum.energy += vertex.throughput * emission;
                        debug_assert!(!sum.energy.is_nan());
                    } else {
                        let hit_primitive = self.world.get_primitive(hit.instance_id);
                        // // println!("{:?}", hit);
                        let pdf = hit_primitive.psa_pdf(
                            prev_vertex.normal * (hit.point - prev_vertex.point).normalized(),
                            prev_vertex.point,
                            hit.point,
                        );
                        let weight = power_heuristic(prev_vertex.pdf_forward, pdf.0);
                        debug_assert!(!pdf.is_nan() && !weight.is_nan(), "{:?}, {}", pdf, weight);
                        sum.energy += vertex.throughput * emission * weight;
                        debug_assert!(!sum.energy.is_nan());
                    }
                }

                if self.light_samples > 0 {
                    let light_contribution = self.estimate_direct_illumination_with_loop(
                        sum.lambda,
                        &hit,
                        &frame,
                        wi,
                        material,
                        vertex.throughput,
                        sampler,
                        &mut profile,
                    );
                    // println!("light contribution: {:?}", light_contribution);
                    sum.energy += light_contribution / (self.light_samples as f32);
                    debug_assert!(
                        !sum.energy.is_nan(),
                        "{:?} {:?}",
                        light_contribution,
                        self.light_samples
                    );
                }
            }
            if self.only_direct {
                break;
            }
        }

        XYZColor::from(sum)
    }
}
