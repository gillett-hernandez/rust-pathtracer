use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::*;
use crate::integrator::*;
use crate::material::Material;
use crate::materials::{MaterialEnum, MaterialId};
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
) {
    let (camera, camera_id, camera_pick_pdf) = world
        .pick_random_camera(camera_pick)
        .expect("camera pick failed");
    if let Some(camera_surface) = camera.get_surface() {
        // let (direction, camera_pdf) = camera_surface.sample(camera_direction_sample, hit.point);
        // let direction = direction.normalized();
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
        if veach_v(&world, point_on_lens, hit.point) {
            let scatter_pdf_into_camera = material.value(&hit, wi, camera_wo);
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
                let ret = (Sample::LightSample(sw, uv), camera_id as u8);
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
        _camera_sample: (Ray, CameraId),
        mut samples: &mut Vec<(Sample, CameraId)>,
    ) -> SingleWavelength {
        // setup: decide light, decide wavelength, emit ray from light, connect light ray vertices to camera.
        let wavelength_sample = sampler.draw_1d();
        let light_pick_sample = sampler.draw_1d();

        let env_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;
        let mut light_g_term: f32 = 1.0;

        let (light_pick_sample, sample_world) =
            light_pick_sample.choose(env_sampling_probability, true, false);
        if !sample_world {
            if self.world.lights.len() > 0 {
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
                let world_aabb = self.world.accelerator.aabb();
                let world_radius = (world_aabb.max - world_aabb.min).0.abs().max_element() / 2.0;
                // println!("sampling world, world radius is {}", world_radius);
                // println!("sampled light emission in world light branch");
                sampled = self.world.environment.sample_emission(
                    world_radius,
                    sampler.draw_2d(),
                    sampler.draw_2d(),
                    VISIBLE_RANGE,
                    wavelength_sample,
                );
            }
        } else {
            if env_sampling_probability > 0.0 {
                // sample world env
                // println!("sampled light emission in world light branch");
                // println!("sampling world, world radius is {}", world_radius);
                let world_radius = self.world.get_world_radius();
                sampled = self.world.environment.sample_emission(
                    world_radius,
                    sampler.draw_2d(),
                    sampler.draw_2d(),
                    VISIBLE_RANGE,
                    wavelength_sample,
                );
                light_g_term = 1.0;
            // sampled = (tmp_sampled.0, tmp_sampled.1, tmp_sampled.2);
            } else {
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
            }
        };
        let mut ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;
        if radiance.0 == 0.0 {
            return SingleWavelength::BLACK;
        }
        let light_pdf = sampled.2;
        let lambda_pdf = sampled.3;

        // let mut beta = SingleEnergy::ONE;
        // let mut beta = radiance / (sampled.2).0;
        let mut beta = radiance;
        let mut beta_pdf = PDF::from(1.0);
        // camera_vertex.lambda = lambda;

        let mut last_bsdf_pdf = PDF::from(0.0);
        // light loop here
        for bounce_count in 0..self.max_bounces {
            if let Some(mut hit) = self.world.hit(ray, INTERSECTION_TIME_OFFSET, INFINITY) {
                // hit some bsdf surface
                // debug_assert!(hit.point.0.is_finite().all(), "ray {:?}, {:?}", ray, hit);
                // println!("whatever1");
                hit.lambda = lambda;
                hit.transport_mode = TransportMode::Radiance;

                debug_assert!(lambda > 0.0);
                let frame = TangentFrame::from_normal(hit.normal);
                let wi = frame.to_local(&-ray.direction).normalized();
                // println!("{:?}. wi {:?} ", hit, wi);

                if bounce_count == 0 {
                    // include first P_A term correctly
                    beta.0 *= light_pdf.0 / lambda_pdf.0 * light_g_term * wi.z()
                        / (hit.point - ray.origin).norm_squared();
                }

                let material = self.world.get_material(hit.material);

                if let MaterialId::Camera(camera_id) = hit.material {
                    // consider the case when the camera is hit directly
                    // print!("checking camera id {} due to hit {:?} ", camera_id, hit);
                    let camera = self.world.get_camera(camera_id as usize);
                    // if we hit it, then it has to have a surface
                    let camera_surface = camera.get_surface().unwrap();

                    let backwards_ray = Ray::new(hit.point, -ray.direction);
                    let pixel_uv = camera.get_pixel_for_ray(backwards_ray);
                    // println!("pixel uv for ray {:?} was {:?}", ray, pixel_uv);
                    if let Some(uv) = pixel_uv {
                        if last_bsdf_pdf.0 <= 0.0 || self.camera_samples == 0 {
                            // not dividing by pdf because it was already handled in the prior iteration loop.
                            let energy = beta;
                            debug_assert!(energy.0.is_finite());
                            let sw = SingleWavelength::new(lambda, energy);
                            let ret = (Sample::LightSample(sw, uv), camera_id);
                            // println!("adding camera sample to splatting list");
                            samples.push(ret);
                        } else {
                            // let hit_primitive = self.world.get_primitive(hit.instance_id);
                            // println!("{:?}", ray.origin - hit.point);

                            let pdf = camera_surface.pdf(hit.normal, ray.origin, hit.point);
                            let weight = power_heuristic(last_bsdf_pdf.0, pdf.0);
                            debug_assert!(
                                !pdf.is_nan() && !weight.is_nan(),
                                "{:?}, {:?}, {}",
                                last_bsdf_pdf,
                                pdf,
                                weight
                            );
                            // not dividing by pdf because it was already handled in the prior iteration loop.
                            let energy = beta * weight;
                            debug_assert!(energy.0.is_finite());
                            let sw = SingleWavelength::new(lambda, energy);
                            let ret = (Sample::LightSample(sw, uv), camera_id);
                            // println!("adding camera sample to splatting list");
                            samples.push(ret);
                        }
                    }
                }

                // // wo is generated in tangent space.
                let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);

                // attempt to connect to camera
                for _ in 0..self.camera_samples {
                    let camera_pick = sampler.draw_1d();
                    // evaluate_direct_importance(&self.world, camera_pick, &mut samples);
                    evaluate_direct_importance(
                        &self.world,
                        camera_pick,
                        sampler.draw_2d(),
                        lambda,
                        beta,
                        material,
                        wi,
                        &hit,
                        &frame,
                        &mut samples,
                    );
                }

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
                    last_bsdf_pdf = pdf;

                    // add normal to avoid self intersection
                    // also convert wo back to world space when spawning the new ray
                    // println!("whatever!!");
                    ray = Ray::new(
                        hit.point
                            + hit.normal * NORMAL_OFFSET * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                        frame.to_world(&wo).normalized(),
                    );
                } else {
                    break;
                }
            }
        }
        SingleWavelength::BLACK
    }
}
