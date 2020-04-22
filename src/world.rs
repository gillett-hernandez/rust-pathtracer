
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
    pub fn pick_random_light(&self, s: &mut Box<dyn Sampler>) -> Option<&Box<dyn Hittable>> {
        let length = self.lights.len();
        if length == 0 {
            None
        } else {
            let x = s.draw_1d().x;
            let idx = (length as f32 * x).clamp(0.0, length as f32 - 1.0) as usize;
            assert!(
                idx < self.lights.len(),
                "{}, {}, {}, {}",
                x,
                length as f32 * x,
                idx,
                length as usize
            );
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
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, f32) {
        unimplemented!()
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> f32 {
        unimplemented!()
    }
    fn get_instance_id(&self) -> usize {
        unimplemented!()
    }
}
