use super::world::World;
use crate::hittable::Hittable;
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
                    // RGBColor::new(1.0, 1.0, 1.0)
                    // RGBColor::new(
                    //     (1.0 + hit.normal.x) / 2.0,
                    //     (1.0 + hit.normal.y) / 2.0,
                    //     (1.0 + hit.normal.z) / 2.0,
                    // )
                    let id = match (hit.material) {
                        Some(id) => id as usize,
                        None => 0,
                    };
                    let cos_i = ray.direction.normalized() * hit.normal.normalized();
                    let material = &self.world.materials[id as usize];
                    let bounce = material.generate(Sample2D::new_random_sample(), ray.direction);
                    let emission: RGBColor = material.emission(ray.direction, bounce);
                    let pdf = material.value(ray.direction, bounce);
                    color += beta * emission;
                    beta *= material.f(ray.direction, bounce) * cos_i / pdf;
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

// vec3 color(ray &r, int depth, long *bounce_count, path *_path, bool skip_light_hit = false) = 0;
