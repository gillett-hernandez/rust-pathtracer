use crate::hittable::*;
use crate::materials::MaterialTable;
use crate::math::*;

pub struct World {
    pub bvh: Box<dyn Indexable>,
    pub lights: Vec<usize>,
    pub materials: MaterialTable,
    pub background: RGBColor,
}

impl World {
    pub fn pick_random_light(&self, s: &Box<dyn Sampler>) -> &Box<dyn Hittable> {
        let index = (s.draw_1d().x * (self.lights.len() as f32)) as usize;
        self.bvh.get_primitive(index)
    }
}

impl Hittable for World {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        self.bvh.hit(r, t0, t1)
    }
    fn sample(&self, s: &Box<dyn Sampler>, point: Point3) -> Vec3 {
        unimplemented!();
    }
    fn pdf(&self, point: Point3, wi: Vec3) -> f32 {
        unimplemented!();
    }
}
