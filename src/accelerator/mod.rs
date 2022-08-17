mod bvh;
mod lbvh;

pub use bvh::{BHShape, BVHNode, BoundingHierarchy, BVH};
pub use lbvh::FlatBVH;
use math::prelude::Ray;

use crate::aabb::*;
use crate::geometry::Instance;
use crate::hittable::{HitRecord, Hittable};
use crate::prelude::InstanceId;

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
                let mut possible_hit_instances = bvh.traverse(&r, instances);
                // possible_hit_instances.sort_unstable_by(|a, b| {
                //     // let hit0_t1 = a.2;
                //     // let hit1_t1 = b.2;
                //     // let sign = (hit1_t1-hit0_t1).signum();
                //     (a.2).partial_cmp(&b.2).unwrap()
                // });
                // let mut closest_so_far: f32 = t1;
                // debug_assert!(!t1.is_nan());
                // let mut hit_record: Option<HitRecord> = None;
                // for (instance, t0_aabb_hit, t1_aabb_hit) in possible_hit_instances {
                //     if t1_aabb_hit < t0 || t0_aabb_hit > t1 {
                //         // if bounding box hit was outside of hit time bounds
                //         continue;
                //     }
                //     if t0_aabb_hit > closest_so_far {
                //         // ignore aabb hit that happened after closest so far
                //         continue;
                //     }
                //     // let t0 = t0.max(t0_aabb_hit);

                //     // let t1 = closest_so_far.min(t1_aabb_hit);
                //     // let tmp_hit_record = instance.hit(r, t0, t1);
                //     let tmp_hit_record = instance.hit(r, t0, closest_so_far);
                //     if let Some(hit) = &tmp_hit_record {
                //         closest_so_far = hit.time;
                //         hit_record = tmp_hit_record;
                //     } else {
                //         continue;
                //     }
                // }

                // temporary inefficient method. brute force somewhat.
                let mut hit_record = None;
                let mut closest_so_far = 0.0;
                for (instance, t0_aabb_hit, t1_aabb_hit) in possible_hit_instances {
                    if t1_aabb_hit < t0 || t0_aabb_hit > t1 {
                        // if bounding box hit was outside of prescribed hit time bounds
                        continue;
                    }

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
    pub fn get_primitive(&self, index: InstanceId) -> &Instance {
        match self {
            Accelerator::List { instances } => instances.get(index as usize).unwrap(),
            Accelerator::BVH { instances, bvh: _ } => instances.get(index as usize).unwrap(),
        }
    }
}
