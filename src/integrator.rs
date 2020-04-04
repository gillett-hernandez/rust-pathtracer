use super::world::World;
use crate::math::Vec3;

pub trait Color {
    fn color(&self) -> Vec3;
}

pub struct Integrator {
    max_bounces: i16,
    world: World,
    // Config config;
}

impl Color for Integrator {
    fn color(&self) -> Vec3 {
        Vec3::ZERO
    }
}

// vec3 color(ray &r, int depth, long *bounce_count, path *_path, bool skip_light_hit = false) = 0;
