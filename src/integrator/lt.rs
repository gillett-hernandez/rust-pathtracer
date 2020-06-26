use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::camera::Camera;
use crate::hittable::Hittable;
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
    pub cameras: Vec<Box<dyn Camera>>,
    // pub only_direct: bool,
}
/*
impl Integrator for LightTracingIntegrator {
    fn color(&self, sampler: &mut Box<dyn Sampler>, samples: &mut Vec<(SingleWavelength, (usize, usize, usize))){
        // setup: decide light, decide wavelength, emit ray from light, connect light ray vertices to camera or to camera ray hit point.
        // two possible formulations: connecting to camera directly and splatting to the pixel buffer
        // or shooting a camera ray and connecting to its hit point.
        // in this integrator, it's the latter
        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();
        let mut camera_vertex = if let Some(mut hit) = self.world.hit(camera_ray, 0.0, INFINITY) {
            if self.world.instance_is_light(hit.instance_id) {
                let material: &Box<dyn Material> = &self.world.get_material(hit.material);
                let (lambda, _lambda_pdf) = material
                    .sample_emission_spectra(hit.uv, VISIBLE_RANGE, wavelength_sample)
                    .expect("instance marked as light did not have any emission spectra");
                hit.lambda = lambda;
                let frame = TangentFrame::from_normal(hit.normal);
                let wi = frame.to_local(&-camera_ray.direction).normalized();
                let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);

                let energy = material.emission(&hit, wi, maybe_wo);
                // energy.0 = 0.0;
                return SingleWavelength { lambda, energy };
            }
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

            let (world_emission, _pdf) = world_emission.expect("world env could not be sampled");
            return world_emission;
            // return world_emission.with_energy(world_emission.energy / pdf.0);
        };

        let camera_hit_frame = TangentFrame::from_normal(camera_vertex.normal);
        let camera_vertex_material: &Box<dyn Material> =
            self.world.get_material(camera_vertex.material);

        let scene_light_sampling_probability = 0.8;

        let sampled;
        let mut light_g_term: f32 = 1.0;

        if self.world.lights.len() > 0 && light_pick_sample.x < scene_light_sampling_probability {
            light_pick_sample.x =
                (light_pick_sample.x / scene_light_sampling_probability).clamp(0.0, 1.0);
            let (light, pick_pdf) = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal, area_pdf) =
                light.sample_surface(sampler.draw_2d());

            let mat_id = light.get_material_id();
            let material: &Box<dyn Material> = &self.world.get_material(mat_id);
            // println!("sampled light emission in instance light branch");
            let tmp_sampled = material
                .sample_emission(
                    light_surface_point,
                    light_surface_normal,
                    VISIBLE_RANGE,
                    sampler.draw_2d(),
                    wavelength_sample,
                )
                .unwrap();
            light_g_term = (light_surface_normal * (&tmp_sampled.0).direction).abs();
            sampled = (
                tmp_sampled.0,
                tmp_sampled.1,
                tmp_sampled.2 * pick_pdf * area_pdf,
            );
        } else {
            light_pick_sample.x = ((light_pick_sample.x - scene_light_sampling_probability)
                / (1.0 - scene_light_sampling_probability))
                .clamp(0.0, 1.0);
            // sample world env
            let world_aabb = self.world.accelerator.bounding_box();
            let world_radius = (world_aabb.max - world_aabb.min).0.abs().max_element() / 2.0;
            // println!("sampled light emission in world light branch");
            sampled = self.world.environment.sample_emission(
                world_radius,
                sampler.draw_2d(),
                VISIBLE_RANGE,
                wavelength_sample,
            );
        };
        let mut ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;
        let light_pdf = sampled.2;
        // let mut beta = SingleEnergy::ONE;
        // let mut beta = radiance / (sampled.2).0;
        let mut beta = radiance;
        let mut beta_pdf = PDF::from(1.0);
        camera_vertex.lambda = lambda;
        let mut sum = SingleWavelength {
            lambda,
            energy: SingleEnergy::ZERO,
        };
        assert!(radiance.0 > 0.0, "radiance was 0, {}", lambda);

        // let mut last_bsdf_pdf = PDF::from(0.0);
        // light loop here
        for bounce_count in 0..self.max_bounces {
            if let Some(mut hit) = self.world.hit(ray, 0.0, INFINITY) {
                // hit some bsdf surface
                // debug_assert!(hit.point.0.is_finite().all(), "ray {:?}, {:?}", ray, hit);
                // println!("whatever1");
                hit.lambda = lambda;

                assert!(lambda > 0.0);
                let frame = TangentFrame::from_normal(hit.normal);
                let wi = frame.to_local(&-ray.direction).normalized();
                // println!("{:?}. wi {:?} ", hit, wi);

                if bounce_count == 0 {
                    // include first P_A term correctly
                    beta.0 *= light_pdf.0 * light_g_term * wi.z()
                        / (hit.point - ray.origin).norm_squared();
                }
                // assert!(
                //     wi.z() > 0.0,
                //     "point: {:?}, normal {:?}, incoming: {:?}, in local space: {:?}",
                //     hit.point,
                //     hit.normal,
                //     -ray.direction,
                //     wi
                // );

                let material: &Box<dyn Material> = &self.world.get_material(hit.material);

                // attempt to connect to camera vertex.

                // if let Some(mut connection_hit) = self.world.hit(lp_to_cv, 0.0, INFINITY) {
                // connection_hit.lambda = lambda;
                // connection succeeded?
                // this emulates g function

                let tmax = (camera_vertex.point - hit.point).norm() * 0.98;
                let lp_to_cv = Ray::new_with_time_and_tmax(
                    hit.point,
                    (camera_vertex.point - hit.point).normalized(),
                    0.0,
                    tmax,
                );

                // this conditional serves as the V term, checking if the camera vertex and light vertex can see each other
                if self.world.hit(lp_to_cv, 0.01, tmax).is_none() {
                    let lp_wi = wi;
                    let lp_wo = frame.to_local(&lp_to_cv.direction);
                    let lp_reflectance = material.f(&hit, lp_wi, lp_wo);
                    let lp_bsdf_pdf = material.value(&hit, lp_wi, lp_wo);
                    let lp_cos_i = lp_wo.z();

                    let cv_wi = camera_hit_frame.to_local(&-camera_ray.direction);
                    let cv_wo = camera_hit_frame.to_local(&-lp_to_cv.direction);
                    let cv_reflectance = camera_vertex_material.f(&camera_vertex, cv_wi, cv_wo);
                    let cv_bsdf_pdf = camera_vertex_material.value(&camera_vertex, cv_wi, cv_wo);
                    let cv_cos_i = cv_wo.z();

                    // G term in veaches paper, except with the V term factored out.
                    let g =
                        // ((camera_vertex.point - hit.point).norm_squared() * cv_cos_i * lp_cos_i)
                        (cv_cos_i * lp_cos_i / (camera_vertex.point - hit.point).norm_squared())
                            .abs();

                    let cst = lp_reflectance * g * cv_reflectance;

                    if cv_bsdf_pdf.0 * lp_bsdf_pdf.0 > 0.0 {
                        sum.energy += radiance * beta * cst;
                    }
                }
                // }

                // // wo is generated in tangent space.
                let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);
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

                if let Some(wo) = maybe_wo {
                    let pdf = material.value(&hit, wi, wo);
                    debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                    if pdf.0 < 0.00000001 || pdf.is_nan() {
                        break;
                    }
                    if self.russian_roulette {
                        // let attenuation = Vec3::from(beta).norm();
                        let attenuation = beta.0;
                        if attenuation < 1.0 && 0.001 < attenuation {
                            if sampler.draw_1d().x > attenuation {
                                break;
                            }

                            beta = beta / attenuation;
                            debug_assert!(!beta.0.is_nan(), "{}", attenuation);
                        }
                    }
                    let cos_i = wo.z();

                    let f = material.f(&hit, wi, wo);
                    beta *= f * cos_i.abs() / pdf.0;
                    // beta *= f * wi.z() * cos_i.abs() / pdf;
                    // beta *= f / pdf;
                    beta_pdf = PDF::from(beta_pdf.0 * pdf.0 * wi.z() * cos_i.abs());
                    debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);
                    // last_bsdf_pdf = pdf;

                    // add normal to avoid self intersection
                    // also convert wo back to world space when spawning the new ray
                    // println!("whatever!!");
                    ray = Ray::new(
                        hit.point + hit.normal * 0.001 * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                        frame.to_world(&wo).normalized(),
                    );
                } else {
                    break;
                }
            }
            // else {
            //     // nothing else to do except return black.
            //     // camera ray hitting the env was already handled
            //     // no lights already handled
            //     // env sampling handled
            //     // and if the light ray doesn't hit anything then what can you do?
            //     sum;
            // }
        }
        assert!(sum.lambda > 0.0, "{:?}", sum);
        // println!("{:?}", sum);
        sum
    }
}
*/
