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
        let mut ray = r;
        let mut color: RGBColor = RGBColor::ZERO;
        let mut beta = RGBColor::new(1.0, 1.0, 1.0);
        for _ in 0..self.max_bounces {
            match (self.world.hit(ray, 0.0, INFINITY)) {
                Some(hit) => {
                    let id = match (hit.material) {
                        Some(id) => id as usize,
                        None => 0,
                    };
                    let cos_i = ray.direction.normalized() * hit.normal.normalized();
                    let material: &Box<dyn Material> = &self.world.materials[id as usize];
                    let bounce =
                        material.generate(&hit, Sample2D::new_random_sample(), ray.direction);
                    let emission = material.emission(&hit, ray.direction, bounce);
                    let pdf = material.value(&hit, ray.direction, bounce);
                    assert!(pdf >= 0.0, "{}", pdf);
                    color += beta * emission;
                    if bounce.norm() < 0.0000001 || pdf < 0.000001 {
                        break;
                    }
                    let bounce = bounce.normalized();
                    beta *= material.f(&hit, ray.direction, bounce) * cos_i / pdf;
                    ray = Ray::new(hit.point, bounce);
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
