mod bvh;
mod lbvh;

pub use bvh::{BHShape, BVHNode, BoundingHierarchy, BVH};
pub use lbvh::FlatBVH;

use crate::aabb::*;
use crate::geometry::Instance;
use crate::hittable::{HitRecord, Hittable};
use crate::math::*;

#[derive(Clone, Copy, Debug)]
pub enum AcceleratorType {
    List,
    BVH,
}

#[derive(Clone, Debug)]
pub enum Accelerator {
    List {
        instances: Vec<Instance>,
    },
    BVH {
        instances: Vec<Instance>,
        bvh: FlatBVH,
    },
}

impl Accelerator {
    pub fn new(mut list: Vec<Instance>, accelerator_type: AcceleratorType) -> Self {
        match accelerator_type {
            AcceleratorType::List => Accelerator::List { instances: list },
            AcceleratorType::BVH => {
                let root = FlatBVH::build(list.as_mut_slice());
                Accelerator::BVH {
                    instances: list,
                    bvh: root,
                }
            }
        }
    }
    pub fn rebuild(&mut self) {
        match self {
            Accelerator::List { instances: _ } => {}
            Accelerator::BVH {
                ref mut instances,
                ref mut bvh,
            } => {
                *bvh = BVH::build(instances.as_mut_slice()).flatten();
            }
        }
    }
}

impl HasBoundingBox for Accelerator {
    fn aabb(&self) -> AABB {
        match self {
            Accelerator::List { instances } => {
                let mut bounding_box: Option<AABB> = None;
                for instance in instances {
                    if (&bounding_box).is_none() {
                        bounding_box = Some(instance.aabb());
                    } else {
                        bounding_box = Some(bounding_box.unwrap().expand(&instance.aabb()));
                    }
                }
                bounding_box.unwrap()
            }
            Accelerator::BVH { instances, bvh: _ } => {
                let mut bounding_box: Option<AABB> = None;
                for instance in instances {
                    if (&bounding_box).is_none() {
                        bounding_box = Some(instance.aabb());
                    } else {
                        bounding_box = Some(bounding_box.unwrap().expand(&instance.aabb()));
                    }
                }
                bounding_box.unwrap()
            }
        }
    }
}

impl Accelerator {
    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        match self {
            Accelerator::List { instances } => {
                // let mut hit_anything = false;
                let mut closest_so_far: f32 = t1;
                let mut hit_record: Option<HitRecord> = None;
                for instance in instances {
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
            Accelerator::BVH { instances, bvh } => {
                let possible_hit_instances = bvh.traverse(&r, instances);
                let mut closest_so_far: f32 = t1;
                debug_assert!(!t1.is_nan());
                let mut hit_record: Option<HitRecord> = None;
                for instance in possible_hit_instances {
                    let tmp_hit_record = instance.hit(r, t0, closest_so_far);
                    if let Some(hit) = &tmp_hit_record {
                        closest_so_far = hit.time;
                        debug_assert!(!closest_so_far.is_nan(), "{:?}", hit);
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
        match self {
            Accelerator::List { instances } => instances.get(index).unwrap(),
            Accelerator::BVH { instances, bvh: _ } => instances.get(index).unwrap(),
        }
    }
}
