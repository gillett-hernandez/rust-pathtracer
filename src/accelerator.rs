use crate::aabb::*;
use crate::geometry::Instance;
use crate::hittable::{HitRecord, Hittable};
use crate::math::*;

pub enum AcceleratorType {
    // BVH,
    List,
}

pub struct Accelerator {
    pub instances: Vec<Instance>,
    pub accelerator_type: AcceleratorType,
}

// pub struct HittableList {
//     pub list: Vec<Box<dyn Hittable>>,
// }

impl Accelerator {
    pub fn new(list: Vec<Instance>, accel_type: AcceleratorType) -> Self {
        // match accel_type {
        //     AcceleratorType::BVH => unimplemented!(),
        //     AcceleratorType::List =>
        // }
        Accelerator {
            instances: list,
            accelerator_type: accel_type,
        }
    }
}

impl HasBoundingBox for Accelerator {
    fn bounding_box(&self) -> AABB {
        match self.accelerator_type {
            AcceleratorType::List => {
                let mut bounding_box: Option<AABB> = None;
                for instance in &self.instances {
                    if (&bounding_box).is_none() {
                        bounding_box = Some(instance.bounding_box());
                    } else {
                        bounding_box = Some(bounding_box.unwrap().expand(instance.bounding_box()));
                    }
                }
                bounding_box.unwrap()
            } // implement BVH
        }
    }
}

impl Accelerator {
    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        match self.accelerator_type {
            AcceleratorType::List => {
                // let mut hit_anything = false;
                let mut closest_so_far: f32 = t1;
                let mut hit_record: Option<HitRecord> = None;
                for instance in &self.instances {
                    // if !instance.bounding_box().hit(r, t0, closest_so_far) {
                    //     continue;
                    // }
                    let tmp_hit_record = instance.hit(r, t0, closest_so_far);
                    if let Some(hit) = &tmp_hit_record {
                        closest_so_far = hit.time;
                        hit_record = tmp_hit_record;
                    } else {
                        continue;
                    }
                }
                hit_record
            }
        }
    }
}

impl Accelerator {
    pub fn get_primitive(&self, index: usize) -> &Instance {
        &self.instances[index]
    }
}
