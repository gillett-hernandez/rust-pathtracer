use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
// use std::f32::INFINITY;
use std::sync::Arc;

use crate::integrator::Integrator;

pub struct LightTracingIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub light_samples: u16,
    pub only_direct: bool,
}

impl Integrator for LightTracingIntegrator {
    fn color(&self, mut sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength {
        // setup: decide light, decide wavelength, emit ray from light, connect light ray vertices to camera or to camera ray hit point.
        // two possible formulations: connecting to camera directly and splatting to the pixel buffer
        // or shooting a camera ray and connecting to its hit point.
        // in this integrator, it's the latter
        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();
        let camera_vertex = if let Some(hit) = self.world.hit(camera_ray, 0.0, INFINITY) {
            hit
        } else {
            let unit_direction = camera_ray.direction.normalized();
            // get phi and theta values for that direction, then convert to UV values for an environment map.
            let u = (PI + unit_direction.y().atan2(unit_direction.x())) / (2.0 * PI);
            let v = unit_direction.z().acos() / PI;

            let world_emission =
                self.world
                    .environment
                    .sample_spd((u, v), VISIBLE_RANGE, wavelength_sample);

            return world_emission.unwrap_or(SingleWavelength::BLACK);
        };

        let env_sampling_probability = 0.5;

        let sampled;

        if self.world.lights.len() > 0 && light_pick_sample.x < env_sampling_probability {
            light_pick_sample.x = (light_pick_sample.x / env_sampling_probability).clamp(0.0, 1.0);
            let light = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal) =
                light.sample_surface(sampler.draw_2d());
            let mat_id = match light.get_material_id() {
                Some(id) => id as usize,
                None => 0,
            };
            let material: &Box<dyn Material> = &self.world.materials[mat_id as usize];
            sampled = material
                .sample_emission(
                    light_surface_point,
                    light_surface_normal,
                    VISIBLE_RANGE,
                    sampler.draw_2d(),
                    wavelength_sample,
                )
                .unwrap();
        } else {
            light_pick_sample.x =
                (light_pick_sample.x / env_sampling_probability - 1.0).clamp(0.0, 1.0);
            // sample world env
            let Sample2D { x: u, y: v } = sampler.draw_2d();
            let world_aabb = self.world.accelerator.bounding_box();
            let world_radius = (world_aabb.max - world_aabb.min).0.abs().max_element() / 2.0;
            sampled = self.world.environment.sample_emission(
                world_radius,
                (u, v),
                VISIBLE_RANGE,
                wavelength_sample,
            );
        };
        let mut light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let mut beta = sampled.1.energy;

        let mut last_bsdf_pdf = 0.0;
        // light loop here
        for _ in 0..self.max_bounces {
            if let Some(hit) = self.world.hit(light_ray, 0.0, INFINITY) {
                // hit some bsdf surface
                // debug_assert!(hit.point.0.is_finite().all(), "ray {:?}, {:?}", ray, hit);
                // println!("whatever1");
                // hit.lambda = lambda;
                // let id = match hit.material {
                //     Some(id) => id as usize,
                //     None => 0,
                // };
                // let frame = TangentFrame::from_normal(hit.normal);
                // let wi = frame.to_local(&-ray.direction).normalized();
                // // assert!(
                // //     wi.z() > 0.0,
                // //     "point: {:?}, normal {:?}, incoming: {:?}, in local space: {:?}",
                // //     hit.point,
                // //     hit.normal,
                // //     -ray.direction,
                // //     wi
                // // );

                // let material: &Box<dyn Material> = &self.world.materials[id as usize];

                // // wo is generated in tangent space.
                // let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);
                // let emission = material.emission(&hit, wi, maybe_wo);

                // if emission.0 > 0.0 {
                //     // check stuff here
                //     if last_bsdf_pdf <= 0.0 || self.light_samples == 0 {
                //         sum.energy += beta * emission;
                //         assert!(!sum.energy.is_nan());
                //     } else {
                //         let hit_primitive = self.world.get_primitive(hit.instance_id);
                //         // // println!("{:?}", hit);
                //         let pdf = hit_primitive.pdf(hit.normal, ray.origin, hit.point);
                //         let weight = power_heuristic(last_bsdf_pdf, pdf);
                //         assert!(!pdf.is_nan() && !weight.is_nan(), "{}, {}", pdf, weight);
                //         sum.energy += beta * emission * weight;
                //         assert!(!sum.energy.is_nan());
                //     }
                // }
                // let mut light_contribution = SingleEnergy::ZERO;
                // let mut _successful_light_samples = 0;
                // for _i in 0..self.light_samples {
                //     if let Some(light) = self.world.pick_random_light(sampler.draw_1d()) {
                //         // determine pick pdf
                //         // as of now the pick pdf is just num lights, however if it were to change this would be where it should change.
                //         let pick_pdf = self.world.lights.len() as f32;
                //         // sample the primitive from hit_point
                //         let (direction, light_pdf) = light.sample(&mut sampler, hit.point);
                //         assert!(light_pdf.is_finite());
                //         if light_pdf == 0.0 {
                //             continue;
                //         }
                //         // direction is already in world space.
                //         // direction is also oriented away from the shading point already, so no need to negate directions until later.
                //         let wo = frame.to_local(&direction);
                //         let light_ray = Ray::new_with_time(
                //             hit.point + hit.normal * 0.01,
                //             direction,
                //             ray.time,
                //             // hit.time,
                //         );
                //         // note: time was changed to ray.time. change to hit.time?
                //         // since direction is already in world space, no need to call frame.to_world(direction) in the above line
                //         let reflectance = material.f(&hit, wi, wo);
                //         // if reflectance.0 < 0.00001 {
                //         //     // if reflectance is 0 for all components, skip this light sample
                //         //     continue;
                //         // }
                //         let dropoff = wo.z().max(0.0);
                //         if dropoff == 0.0 {
                //             continue;
                //         }
                //         // let dropoff = wo.z().abs();
                //         if let Some(mut light_hit) = self.world.hit(light_ray, 0.0, INFINITY) {
                //             light_hit.lambda = sum.lambda;
                //             // note: changed t0 to 0.0. change back to hit.time maybe?
                //             //
                //             // maybe if the instance that was hit was a light as well, redo the sampling calculations for that light instead?
                //             // let light_pdf =
                //             //     light.pdf(light_hit.normal, hit.point, light_hit.point);
                //             let scatter_pdf_for_light_ray = material.value(&hit, wi, wo);
                //             let weight = power_heuristic(light_pdf, scatter_pdf_for_light_ray);
                //             if light_hit.instance_id == light.get_instance_id() {
                //                 let emission_material =
                //                     &self.world.materials[light_hit.material.unwrap() as usize];
                //                 let light_wi = TangentFrame::from_normal(light_hit.normal)
                //                     .to_local(&-direction);
                //                 let sampled_light_emission =
                //                     emission_material.emission(&light_hit, light_wi, None);
                //                 assert!(sampled_light_emission.0 >= 0.0);
                //                 // successful_light_samples += 1;
                //                 light_contribution +=
                //                     reflectance * beta * dropoff * sampled_light_emission * weight
                //                         / light_pdf
                //                         / pick_pdf;
                //                 assert!(
                //                     !light_contribution.0.is_nan(),
                //                     "l {:?} r {:?} b {:?} d {:?} s {:?} w {:?} p {:?} lp {:?}",
                //                     light_contribution,
                //                     reflectance,
                //                     beta,
                //                     dropoff,
                //                     sampled_light_emission,
                //                     weight,
                //                     pick_pdf,
                //                     light_pdf
                //                 );
                //             }
                //         }
                //     } else {
                //         break;
                //     }
                // }
                // if self.light_samples > 0 {
                //     // println!("light contribution: {:?}", light_contribution);
                //     sum.energy += light_contribution / (self.light_samples as f32);
                //     assert!(
                //         !sum.energy.is_nan(),
                //         "{:?} {:?}",
                //         light_contribution,
                //         self.light_samples
                //     );
                // }
                // if self.only_direct {
                //     break;
                // }
                // // println!("whatever!");
                // if let Some(wo) = maybe_wo {
                //     let pdf = material.value(&hit, wi, wo);
                //     debug_assert!(pdf >= 0.0, "pdf was less than 0 {}", pdf);
                //     if pdf < 0.00000001 || pdf.is_nan() {
                //         break;
                //     }
                //     if self.russian_roulette {
                //         // let attenuation = Vec3::from(beta).norm();
                //         let attenuation = beta.0;
                //         if attenuation < 1.0 && 0.001 < attenuation {
                //             if sampler.draw_1d().x > attenuation {
                //                 break;
                //             }

                //             beta = beta / attenuation;
                //             debug_assert!(!beta.0.is_nan(), "{}", attenuation);
                //         }
                //     }
                //     let cos_i = wo.z();

                //     let f = material.f(&hit, wi, wo);
                //     beta *= f * cos_i.abs() / pdf;
                //     debug_assert!(!beta.0.is_nan(), "{:?} {} {}", f, cos_i, pdf);
                //     last_bsdf_pdf = pdf;
                //     // debug_assert!(wi.z() * wo.z() > 0.0, "{:?} {:?}", wi, wo);
                //     // add normal to avoid self intersection
                //     // also convert wo back to world space when spawning the new ray
                //     // println!("whatever!!");
                //     ray = Ray::new(
                //         hit.point + hit.normal * 0.001 * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                //         frame.to_world(&wo).normalized(),
                //     );
                // } else {
                //     break;
                // }
            } else {
                // nothing else to do except return black.
                // camera ray hitting the env was already handled
                // no lights already handled
                // env sampling handled
                // and if the light ray doesn't hit anything then what can you do?
                return SingleWavelength::BLACK;
            }
        }
        SingleWavelength::BLACK
    }
}
