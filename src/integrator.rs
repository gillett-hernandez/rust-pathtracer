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
        let russian_roulette = true;
        let mut ray = r;
        let mut color: RGBColor = RGBColor::ZERO;
        let mut beta = RGBColor::new(1.0, 1.0, 1.0);
        let sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        for _ in 0..self.max_bounces {
            match (self.world.hit(ray, 0.0, INFINITY)) {
                Some(hit) => {
                    let id = match (hit.material) {
                        Some(id) => id as usize,
                        None => 0,
                    };
                    let wi = -ray.direction;
                    let cos_i = wi.normalized() * hit.normal.normalized();
                    let material: &Box<dyn Material> = &self.world.materials[id as usize];
                    let wo = material.generate(&hit, &sampler, wi);
                    let emission = material.emission(&hit, wi, wo);
                    let pdf = material.value(&hit, wi, wo);
                    assert!(pdf >= 0.0, "pdf was less than 0 {}", pdf);
                    color += beta * emission;
                    if pdf < 0.0000001 || wo.norm() < 0.00000001 {
                        break;
                    }
                    if russian_roulette {
                        let attenuation = Vec3::from(beta).norm();
                        if attenuation < 1.0 && 0.001 < attenuation {
                            if sampler.draw_1d().x > attenuation {
                                break;
                            }

                            beta = beta / attenuation;
                        }
                    }
                    let wo = wo.normalized();
                    // beta *= material.f(&hit, wi, wo) * cos_i.abs();
                    beta *= material.f(&hit, wi, wo) * cos_i.abs() / pdf;
                    // add normal to avoid self intersection
                    ray = Ray::new(hit.point + hit.normal * 0.000001, wo);
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
