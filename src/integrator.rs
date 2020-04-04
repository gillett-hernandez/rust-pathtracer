use super::world::World;
use crate::math::{Float, Point3, Ray, Vec3, RGBColor};

pub trait Integrator {
    fn color(&self, r: Ray) -> RGBColor;
}

pub struct PathTracingIntegrator {
    max_bounces: i16,
    world: World,
    // Config config;
}

impl Integrator for PathTracingIntegrator {
    fn color(&self) -> RGBColor {
        RGBColor::ZERO
    }
}

// vec3 color(ray &r, int depth, long *bounce_count, path *_path, bool skip_light_hit = false) = 0;
