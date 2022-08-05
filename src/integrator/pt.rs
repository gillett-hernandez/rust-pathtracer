use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::{random_walk, veach_v, LightSourceType, SurfaceVertex, VertexType};
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};
use crate::math::*;
use crate::world::World;
use crate::world::{TransportMode, NORMAL_OFFSET};

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

const USE_VEACH_V: bool = false;

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
            // TODO: figure out why the hell the USE_VEACH_V branch was so bad.
            if USE_VEACH_V {
                // determine pick pdf
                // as of now the pick pdf is just num lights, however if it were to change this would be where it should change.
                // sample the primitive from hit_point
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

                let cos_i = light_vertex_wi.z().abs();
                if cos_i == 0.0 {
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
                // FIXME: why are we using hit.uv for this, since hit.uv is the uv for the hit on the material, not on the light

                let emission = light_material.emission(
                    hit.lambda,
                    hit.uv,
                    hit.transport_mode,
                    light_vertex_wi,
                );
                // this should be the same as the other method, but maybe not.
                if emission.0 == 0.0 {
                    return SingleEnergy::ZERO;
                }

                profile.shadow_rays += 1;
                if veach_v(&self.world, point_on_light, hit.point) {
                    let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);

                    debug_assert!(emission.0 >= 0.0);
                    // successful_light_samples += 1;
                    let v = reflectance * throughput * cos_i * emission * weight / light_pdf.0;
                    debug_assert!(
                        v.0.is_finite(),
                        "{:?},{:?},{:?},{:?},{:?},{:?},",
                        reflectance,
                        throughput,
                        cos_i,
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
            } else {
                // light direction is in world space, and is from hit.point
                let (light_direction, light_pdf) = light.sample(additional_light_sample, hit.point);
                let light_pdf = light_pdf * light_pick_pdf;

                if light_pdf.0 == 0.0 {
                    return 0.0.into();
                }
                let bsdf_wo = frame.to_local(&light_direction);
                let (reflectance, bounce_pdf) =
                    material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, bsdf_wo);
                let weight = power_heuristic(light_pdf.0, bounce_pdf.0);

                let shadow_ray = Ray::new(
                    hit.point + hit.normal * NORMAL_OFFSET * bsdf_wo.z().signum(),
                    light_direction,
                );

                profile.shadow_rays += 1;
                if let Some(shadow_hit) = self.world.hit(shadow_ray, 0.0, INFINITY) {
                    if matches!(shadow_hit.material, MaterialId::Light(_)) {
                        let light_material = self.world.get_material(shadow_hit.material);
                        let light_local_frame = TangentFrame::from_normal(shadow_hit.normal);
                        let light_local_wi = light_local_frame.to_local(&-light_direction);

                        let light_emission = light_material.emission(
                            hit.lambda,
                            shadow_hit.uv,
                            hit.transport_mode,
                            light_local_wi,
                        );

                        let cos_i = light_local_wi.z().abs();
                        let cos_o = bsdf_wo.z().abs();

                        #[rustfmt::skip]
                        let v = reflectance
                            * throughput
                            * cos_o
                            * cos_i
                            * light_emission
                            * weight
                            / light_pdf.0;
                        // if throughput.0 == 1.0 {
                        //     info!(
                        //         "{:?} = {:?}*{:?}*{:?}*{:?}*{:?}*{:?}/{:?}",
                        //         v,
                        //         throughput,
                        //         reflectance,
                        //         light_emission,
                        //         bsdf_wo,
                        //         light_local_wi,
                        //         weight,
                        //         light_pdf
                        //     );
                        // }
                        debug_assert!(
                            v.0.is_finite(),
                            "{:?},{:?},{:?},{:?},{:?},{:?},{:?},",
                            reflectance,
                            throughput,
                            cos_o,
                            cos_i,
                            light_emission,
                            weight,
                            light_pdf.0
                        );
                        return v;
                    }
                }
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

        let local_cosine_theta = local_wo.z();
        // return 0 if hemisphere doesn't match.
        if local_cosine_theta <= 0.0 {
            return 0.0.into();
        }

        let (reflectance, scatter_pdf_for_light_ray) =
            material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, local_wo);

        profile.shadow_rays += 1;
        // TODO: add support for passthrough material, such that it doesn't fully interfere with direct illumination
        if let Some(_light_hit) = self.world.hit(
            Ray::new(
                hit.point + hit.normal * NORMAL_OFFSET * direction.z().signum(),
                direction,
            ),
            0.0,
            INFINITY,
        ) {
            // TODO: handle case where we intended to hit the world with the shadow ray but instead hit a light.
            0.0.into()

            // light_hit.lambda = lambda;
            // let material = self.world.get_material(light_hit.material);

            // let point_on_light = light_hit.point;
            // let light_frame = TangentFrame::from_normal(light_hit.normal);
            // let light_wi = light_frame.to_local(&-direction);
            // let dropoff = light_wi.z().abs();
            // if dropoff == 0.0 {
            //     return SingleEnergy::ZERO;
            // }
            // // if reflectance.0 < 0.00001 {
            // //     // if reflectance is 0 for all components, skip this light sample
            // //     continue;
            // // }
            // let emission = material.emission(
            //     light_hit.lambda,
            //     light_hit.uv,
            //     light_hit.transport_mode,
            //     light_wi,
            // );
            // if emission.0 > 0.0 {
            //     let light = self.world.get_primitive(light_hit.instance_id);
            //     let pdf = light.psa_pdf(
            //         hit.normal * (point_on_light - hit.point).normalized(),
            //         hit.point,
            //         point_on_light,
            //     );

            //     if pdf.0 == 0.0 {
            //         return SingleEnergy::ZERO;
            //     }
            //     let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);
            //     reflectance * throughput * dropoff * emission * weight / light_pdf.0
            // } else {
            //     SingleEnergy::ZERO
            // }
        } else {
            // successfully world is visible along this ray
            let emission = self.world.environment.emission(uv, lambda);

            // calculate weight
            let weight = power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);
            // include
            let v = weight * throughput * reflectance * emission * local_cosine_theta.abs()
                / light_pdf.0;
            debug_assert!(
                v.0.is_finite(),
                "{:?},{:?},{:?},{:?},{:?},{:?},",
                reflectance,
                local_cosine_theta,
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
        profile: &mut Profile,
    ) -> SingleEnergy {
        let mut light_contribution = SingleEnergy::ZERO;
        let env_sampling_probability = self.world.get_env_sampling_probability();
        if self.world.lights.is_empty() && env_sampling_probability == 0.0 {
            return SingleEnergy::ZERO;
        }
        for _ in 0..self.light_samples {
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
                    profile,
                );
            } else {
                light_contribution += self.estimate_direct_illumination(
                    hit,
                    frame,
                    wi,
                    material,
                    throughput,
                    light_pick_sample,
                    sampler.draw_2d(),
                    profile,
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
        sampler: &mut Box<dyn Sampler>,
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
            camera.sample_we(film_sample, sampler, sum.lambda);
        let camera_pdf = throughput_and_pdf;
        if camera_pdf.0 == 0.0 {
            return XYZColor::BLACK;
        }

        let max_bounces = if self.only_direct {
            1
        } else {
            self.max_bounces
        };
        let mut path: Vec<SurfaceVertex> = Vec::with_capacity(1 + max_bounces as usize);

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
            100.0,
            0.0,
            1.0,
        ));
        let _ = random_walk(
            camera_ray,
            lambda,
            max_bounces,
            SingleEnergy::from(throughput_and_pdf.0),
            TransportMode::Importance,
            sampler,
            &self.world,
            &mut path,
            self.min_bounces,
            profile,
        );
        // sum.energy += additional_contribution.unwrap_or(0.0.into());

        for (index, vertex) in path.iter().enumerate().skip(1) {
            let prev_vertex = path[index - 1];
            // for every vertex past the 1st one (which is on the camera), evaluate the direct illumination at that vertex, and if it hits a light evaluate the added energy
            if let VertexType::LightSource(light_source) = vertex.vertex_type {
                if light_source == LightSourceType::Environment {
                    // ray direction is stored in vertex.normal
                    let wo = vertex.normal;
                    let uv = direction_to_uv(wo);
                    let emission = self.world.environment.emission(uv, lambda);
                    let nee_psa_pdf = self.world.environment.pdf_for(uv); // * (prev_vertex.normal * (-wo)).abs(); // * prev_vertex.local_wo.z();
                    let bsdf_psa_pdf = prev_vertex.pdf_forward;
                    let weight = power_heuristic(bsdf_psa_pdf, nee_psa_pdf.0);

                    profile.env_hits += 1;
                    sum.energy += weight * vertex.throughput * emission;
                    debug_assert!(
                        !sum.energy.is_nan(),
                        "{:?} {:?} {:?}",
                        weight,
                        vertex.throughput,
                        emission
                    );
                } else {
                    // let hit = HitRecord::from(*vertex);

                    let wi = vertex.local_wi;
                    let material = self.world.get_material(vertex.material_id);

                    let emission =
                        material.emission(vertex.lambda, vertex.uv, TransportMode::Importance, wi);

                    if emission.0 > 0.0 {
                        if self.light_samples == 0
                            || matches!(prev_vertex.vertex_type, VertexType::Camera)
                        {
                            sum.energy += vertex.throughput * emission;
                            debug_assert!(!sum.energy.is_nan());
                        } else {
                            let hit_primitive = self.world.get_primitive(vertex.instance_id);

                            let hypothetical_nee_pdf = hit_primitive.psa_pdf(
                                prev_vertex.normal
                                    * (vertex.point - prev_vertex.point).normalized(),
                                prev_vertex.point,
                                vertex.point,
                            );
                            let weight =
                                power_heuristic(prev_vertex.pdf_forward, hypothetical_nee_pdf.0);

                            // info!(
                            //     "{:?}, {:?} ---- {:?}, {:?}, {:?}",
                            //     prev_vertex.pdf_forward,
                            //     hypothetical_nee_pdf.0,
                            //     weight,
                            //     vertex.throughput,
                            //     emission
                            // );

                            debug_assert!(
                                !hypothetical_nee_pdf.is_nan() && !weight.is_nan(),
                                "{:?}, {}",
                                hypothetical_nee_pdf,
                                weight
                            );
                            // NOTE: not dividing by prev_vertex.pdf_forward because vertex.throughput already factors that in, due to how random walk works
                            sum.energy += weight * vertex.throughput * emission;

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
                    panic!(
                        "material should not be emissive, {}, {:?}",
                        material.get_name(),
                        vertex.material_id
                    );
                    // if prev_vertex.pdf_forward <= 0.0 || self.light_samples == 0 {
                    //     sum.energy += vertex.throughput * emission;
                    //     debug_assert!(!sum.energy.is_nan());
                    // } else {
                    //     let hit_primitive = self.world.get_primitive(hit.instance_id);
                    //     // // println!("{:?}", hit);
                    //     let pdf = hit_primitive.psa_pdf(
                    //         prev_vertex.normal * (hit.point - prev_vertex.point).normalized(),
                    //         prev_vertex.point,
                    //         hit.point,
                    //     );
                    //     let weight = power_heuristic(prev_vertex.pdf_forward, pdf.0);
                    //     debug_assert!(!pdf.is_nan() && !weight.is_nan(), "{:?}, {}", pdf, weight);
                    //     sum.energy += weight * vertex.throughput * emission;
                    //     debug_assert!(!sum.energy.is_nan());
                    // }
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
                        profile,
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
        }

        XYZColor::from(sum)
    }
}
