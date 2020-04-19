use super::world::World;
use crate::config::Settings;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::math::*;
use std::f32::INFINITY;
use std::sync::Arc;

pub trait Integrator: Sync + Send {
    fn color(&self, r: Ray) -> RGBColor;
    fn get_world(&self) -> &World;
}

pub struct PathTracingIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub light_samples: u16,
}

impl Integrator for PathTracingIntegrator {
    fn color(&self, r: Ray) -> RGBColor {
        let mut ray = r;
        let mut color: RGBColor = RGBColor::ZERO;
        let mut beta = RGBColor::new(1.0, 1.0, 1.0);
        let sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        for current_bounce in 0..self.max_bounces {
            match (self.world.hit(ray, 0.0, INFINITY)) {
                Some(hit) => {
                    let id = match (hit.material) {
                        Some(id) => id as usize,
                        None => 0,
                    };
                    let frame = TangentFrame::from_normal(hit.normal);
                    let wi = frame.to_local(&-ray.direction).normalized();
                    // assert!(
                    //     wi.z() > 0.0,
                    //     "point: {:?}, normal {:?}, incoming: {:?}, in local space: {:?}",
                    //     hit.point,
                    //     hit.normal,
                    //     -ray.direction,
                    //     wi
                    // );

                    let material: &Box<dyn Material> = &self.world.materials[id as usize];

                    // wo is generated in tangent space.
                    let maybe_wo: Option<Vec3> = material.generate(&hit, &sampler, wi);
                    let emission = material.emission(&hit, wi, maybe_wo);

                    if emission.0.max_element() > 0.0 {
                        color += beta * emission;
                        // check stuff here
                        /*if last_bsdf_pdf <= 0 {
                            color += beta * emission
                        } else {
                            let pdf = hit.primitive.pdf(hit.point, ray.direction);
                            let weight = power_heuristic(last_bsdf_pdf, pdf);
                            color += beta * emission * weight;
                        }*/
                    }
                    let mut light_contribution = RGBColor::ZERO;
                    for i in 0..self.light_samples {
                        if let Some(light) = self.world.pick_random_light(&sampler) {
                            // determine pick pdf
                            // as of now the pick pdf is just 1 / num lights, however if it were to change this would be where it should change.
                            let pick_pdf = self.world.lights.len() as f32;
                            // direction is already in world space.
                            let direction = light.sample(&sampler, hit.point);
                            let pdf = light.pdf(hit.point, direction);
                            let wo = frame.to_local(&direction);
                            let reflectance = material.f(&hit, wi, wo);
                            // since direction is already in world space, no need to call frame.to_world(direction)
                            let light_ray = Ray::new(hit.point + hit.normal * 0.01, direction);
                            let dropoff = (direction * hit.normal).max(0.0);
                            let scatter_pdf_for_light_ray = material.value(&hit, wi, wo);
                            let weight = power_heuristic(pdf, scatter_pdf_for_light_ray);

                            if let Some(light_hit) = self.world.hit(light_ray, hit.time, INFINITY) {
                                // TODO: constrain further by identifying if we hit the primitive we meant to hit. that is, if hit.instance_id == light.instance_id;
                                // maybe if the instance that was hit was a light as well, redo the sampling calculations for that light instead?
                                let emission_material: &Box<dyn Material> =
                                    &self.world.materials[light_hit.material.unwrap() as usize];
                                let light_wi = TangentFrame::from_normal(light_hit.normal)
                                    .to_local(&-direction);
                                let sampled_light_emission =
                                    emission_material.emission(&light_hit, light_wi, None);
                                light_contribution +=
                                    reflectance * beta * dropoff * sampled_light_emission * weight
                                        / pdf
                                        / pick_pdf;
                            }
                        } else {
                            break;
                        }
                    }
                    if let Some(wo) = maybe_wo {
                        let pdf = material.value(&hit, wi, wo);
                        assert!(pdf >= 0.0, "pdf was less than 0 {}", pdf);
                        if pdf < 0.0000001 {
                            break;
                        }
                        if self.russian_roulette {
                            // let attenuation = Vec3::from(beta).norm();
                            let attenuation = Vec3::from(beta).0.max_element();
                            if attenuation < 1.0 && 0.001 < attenuation {
                                if sampler.draw_1d().x > attenuation {
                                    break;
                                }

                                beta = beta / attenuation;
                            }
                        }
                        let cos_i = wo.z();
                        beta *= material.f(&hit, wi, wo) * cos_i.abs() / pdf;
                        debug_assert!(wi.z() * wo.z() > 0.0, "{:?} {:?}", wi, wo);
                        // add normal to avoid self intersection
                        // also convert wo back to world space when spawning the new ray
                        ray = Ray::new(
                            hit.point + hit.normal * 0.001,
                            frame.to_world(&wo).normalized(),
                        );
                    } else {
                        break;
                    }
                }
                None => {
                    color += beta * self.world.background;
                    break;
                }
            }
        }
        color
    }
    fn get_world(&self) -> &World {
        &self.world
    }
}
