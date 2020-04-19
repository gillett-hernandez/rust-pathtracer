use crate::aabb::*;
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
    pub fn pick_random_light(&self, s: &Box<dyn Sampler>) -> Option<&Box<dyn Hittable>> {
        let length = self.lights.len();
        if length == 0 {
            None
        } else {
            let idx = (length as f32 * s.draw_1d().x) as usize;
            Some(self.bvh.get_primitive(self.lights[idx]))
        }
    }

    pub fn get_primitive(&self, index: usize) -> &Box<dyn Hittable> {
        self.bvh.get_primitive(index)
    }
}

impl HasBoundingBox for World {
    fn bounding_box(&self) -> AABB {
        self.bvh.bounding_box()
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
    fn get_instance_id(&self) -> usize {
        unimplemented!();
    }
}
