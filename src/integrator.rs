use super::world::World;
use crate::hittable::Hittable;
// use crate::materials::MaterialTable;
use crate::material::Material;
use crate::math::*;
use std::f32::INFINITY;

pub trait Integrator {
    fn color(&self, r: Ray) -> RGBColor;
    fn get_world(&self) -> &World;
}

pub struct PathTracingIntegrator {
    pub max_bounces: i16,
    pub world: World,
    // Config config;
}

impl Integrator for PathTracingIntegrator {
    fn color(&self, r: Ray) -> RGBColor {
        const RUSSIAN_ROULETTE: bool = false;
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

                    color += beta * emission;
                    if let Some(wo) = maybe_wo {
                        let pdf = material.value(&hit, wi, wo);
                        assert!(pdf >= 0.0, "pdf was less than 0 {}", pdf);
                        if pdf < 0.0000001 {
                            break;
                        }
                        if RUSSIAN_ROULETTE {
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
                        // beta *= material.f(&hit, wi, wo) * cos_i.abs();
                        beta *= material.f(&hit, wi, wo) * cos_i.abs() / pdf;
                        // debug_assert!(wi.z() * wo.z() > 0.0, "{:?} {:?}", wi, wo);
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
                    // color += beta * self.world.background * 2.0 * PI * PI;
                    if current_bounce > 0 {
                        // hit env after bouncing
                        // beta = beta * (4.0 * PI);
                    } else {
                        // hit env straight away
                    }
                    color += beta * self.world.background;
                    // color += beta * self.world.background * 4.0 * PI;
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
