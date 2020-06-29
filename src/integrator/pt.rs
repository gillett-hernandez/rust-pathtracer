use crate::world::World;
// use crate::config::Settings;
use crate::hittable::Hittable;
use crate::integrator::{veach_v, SamplerIntegrator};
use crate::material::Material;
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::NORMAL_OFFSET;

use std::f32::INFINITY;
use std::sync::Arc;

pub struct PathTracingIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub light_samples: u16,
    pub only_direct: bool,
    pub wavelength_bounds: Bounds1D,
}

impl SamplerIntegrator for PathTracingIntegrator {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength {
        let mut ray = camera_ray;
        // println!("{:?}", ray);
        let mut sum = SingleWavelength::new_from_range(sampler.draw_1d().x, VISIBLE_RANGE);
        let mut beta: SingleEnergy = SingleEnergy::ONE;
        let mut last_bsdf_pdf = PDF::from(0.0);

        for _ in 0..self.max_bounces {
            // println!("whatever0");
            match self.world.hit(ray, 0.0, INFINITY) {
                Some(mut hit) => {
                    debug_assert!(hit.point.0.is_finite().all(), "ray {:?}, {:?}", ray, hit);
                    // println!("whatever1");
                    hit.lambda = sum.lambda;
                    let frame = TangentFrame::from_normal(hit.normal);
                    let wi = frame.to_local(&-ray.direction).normalized();
                    // debug_assert!(
                    //     wi.z() > 0.0,
                    //     "point: {:?}, normal {:?}, incoming: {:?}, in local space: {:?}",
                    //     hit.point,
                    //     hit.normal,
                    //     -ray.direction,
                    //     wi
                    // );

                    let material = self.world.get_material(hit.material);

                    // wo is generated in tangent space.
                    let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);
                    let emission = material.emission(&hit, wi, maybe_wo);

                    if emission.0 > 0.0 {
                        // check stuff here
                        if last_bsdf_pdf.0 <= 0.0 || self.light_samples == 0 {
                            sum.energy += beta * emission;
                            debug_assert!(!sum.energy.is_nan());
                        } else {
                            let hit_primitive = self.world.get_primitive(hit.instance_id);
                            // // println!("{:?}", hit);
                            let pdf = hit_primitive.pdf(hit.normal, ray.origin, hit.point);
                            let weight = power_heuristic(last_bsdf_pdf.0, pdf.0);
                            debug_assert!(
                                !pdf.is_nan() && !weight.is_nan(),
                                "{:?}, {}",
                                pdf,
                                weight
                            );
                            sum.energy += beta * emission * weight;
                            debug_assert!(!sum.energy.is_nan());
                        }
                    }
                    let mut light_contribution = SingleEnergy::ZERO;
                    let mut _successful_light_samples = 0;
                    for _i in 0..self.light_samples {
                        if let Some((light, light_pick_pdf)) =
                            self.world.pick_random_light(sampler.draw_1d())
                        {
                            // determine pick pdf
                            // as of now the pick pdf is just num lights, however if it were to change this would be where it should change.
                            // sample the primitive from hit_point
                            // let (direction, light_pdf) = light.sample(sampler.draw_2d(), hit.point);
                            let (point_on_light, normal, light_area_pdf) =
                                light.sample_surface(sampler.draw_2d());
                            debug_assert!(light_area_pdf.0.is_finite());
                            if light_area_pdf.0 == 0.0 {
                                continue;
                            }
                            let direction = (point_on_light - hit.point).normalized();
                            // direction is already in world space.
                            // direction is also oriented away from the shading point already, so no need to negate directions until later.
                            let wo = frame.to_local(&direction);
                            let light_wi =
                                TangentFrame::from_normal(normal).to_local(&(-direction));

                            let dropoff = wo.z().max(0.0);
                            if dropoff == 0.0 {
                                continue;
                            }
                            // since direction is already in world space, no need to call frame.to_world(direction) in the above line
                            let reflectance = material.f(&hit, wi, wo);
                            // if reflectance.0 < 0.00001 {
                            //     // if reflectance is 0 for all components, skip this light sample
                            //     continue;
                            // }

                            let pdf = light.pdf(normal, hit.point, point_on_light);
                            let light_pdf = pdf * light_pick_pdf;
                            if light_pdf.0 == 0.0 {
                                // go to next pick
                                continue;
                            }

                            let light_material = self.world.get_material(light.get_material_id());
                            let emission = light_material.emission(&hit, light_wi, None);
                            // this should be the same as the other method, but maybe not.

                            if veach_v(&self.world, point_on_light, hit.point) {
                                let scatter_pdf_for_light_ray = material.value(&hit, wi, wo);
                                let weight =
                                    power_heuristic(light_pdf.0, scatter_pdf_for_light_ray.0);

                                debug_assert!(emission.0 >= 0.0);
                                // successful_light_samples += 1;
                                light_contribution +=
                                    reflectance * beta * dropoff * emission * weight / light_pdf.0;
                                debug_assert!(
                                    !light_contribution.0.is_nan(),
                                    "l {:?} r {:?} b {:?} d {:?} s {:?} w {:?} p {:?} ",
                                    light_contribution,
                                    reflectance,
                                    beta,
                                    dropoff,
                                    emission,
                                    weight,
                                    light_pdf
                                );
                            }
                        } else {
                            break;
                        }
                    }
                    if self.light_samples > 0 {
                        // println!("light contribution: {:?}", light_contribution);
                        sum.energy += light_contribution / (self.light_samples as f32);
                        debug_assert!(
                            !sum.energy.is_nan(),
                            "{:?} {:?}",
                            light_contribution,
                            self.light_samples
                        );
                    }
                    if self.only_direct {
                        break;
                    }
                    // println!("whatever!");
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
                        beta *= f * cos_i.abs() / pdf.into();
                        debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);
                        last_bsdf_pdf = pdf.into();

                        // add normal to avoid self intersection
                        // also convert wo back to world space when spawning the new ray
                        // println!("whatever!!");
                        ray = Ray::new(
                            hit.point
                                + hit.normal
                                    * NORMAL_OFFSET
                                    * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                            frame.to_world(&wo).normalized(),
                        );
                    } else {
                        break;
                    }
                }
                None => {
                    let unit_direction = ray.direction.normalized();
                    // get phi and theta values for that direction, then convert to UV values for an environment map.
                    let u = (PI + unit_direction.y().atan2(unit_direction.x())) / (2.0 * PI);
                    let v = unit_direction.z().acos() / PI;

                    let world_emission = self.world.environment.emission((u, v), sum.lambda);
                    sum.energy += beta * world_emission;
                    debug_assert!(!sum.energy.is_nan(), "{:?} {:?}", beta, world_emission);
                    break;
                }
            }
        }
        // let xyz_from_sum = XYZColor::from(sum);
        // let rgb_from_xyz = RGBColor::from(xyz_from_sum);

        debug_assert!(!sum.energy.is_nan(), "{:?}", sum);
        sum
    }
}
