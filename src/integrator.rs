use super::world::World;
use crate::hittable::Hittable;
use crate::math::{Point3, RGBColor, Ray, Vec3};
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
        match (self.world.hit(r, 0.0, INFINITY)) {
            Some(hit) => {
                // RGBColor::new(1.0, 1.0, 1.0)
                RGBColor::new(
                    hit.point.x.max(0.0),
                    hit.point.y.max(0.0),
                    hit.point.z.max(0.0),
                )
            }
            None => self.world.background,
        }
    }
    fn get_world(&self) -> &World {
        &self.world
    }
}

// vec3 color(ray &r, int depth, long *bounce_count, path *_path, bool skip_light_hit = false) = 0;
