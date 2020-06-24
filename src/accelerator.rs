use crate::aabb::*;
use crate::geometry::Aggregate;
use crate::hittable::{HitRecord, Hittable, Indexable};
use crate::math::*;

pub enum AcceleratorType {
    // BVH,
    List,
}

pub struct Accelerator {
    pub aggregates: Vec<Aggregate<'static>>,
    pub accelerator_type: AcceleratorType,
}

// pub struct HittableList {
//     pub list: Vec<Box<dyn Hittable>>,
// }

impl Accelerator {
    pub fn new(list: Vec<Aggregate<'static>>, accel_type: AcceleratorType) -> Self {
        // match accel_type {
        //     AcceleratorType::BVH => unimplemented!(),
        //     AcceleratorType::List =>
        // }
        Accelerator {
            aggregates: list,
            accelerator_type: accel_type,
        }
    }
}

impl HasBoundingBox for Accelerator {
    fn bounding_box(&self) -> AABB {
        match self.accelerator_type {
            AcceleratorType::List => {
                let mut bounding_box: Option<AABB> = None;
                for aggregate in &self.aggregates {
                    if (&bounding_box).is_none() {
                        bounding_box = Some(aggregate.bounding_box());
                    } else {
                        bounding_box = Some(bounding_box.unwrap().expand(aggregate.bounding_box()));
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
                for aggregate in &self.aggregates {
                    // if !aggregate.bounding_box().hit(r, t0, closest_so_far) {
                    //     continue;
                    // }
                    let tmp_hit_record = aggregate.hit(r, t0, closest_so_far);
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
    pub fn get_primitive(&self, index: usize) -> &Aggregate {
        &self.aggregates[index]
    }
}
