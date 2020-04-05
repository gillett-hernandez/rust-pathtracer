use crate::hittable::*;
use crate::math::*;

pub struct World {
    pub bvh: Box<dyn Hittable>,
    pub background: RGBColor,
}

impl Hittable for World {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        self.bvh.hit(r, t0, t1)
    }
}
