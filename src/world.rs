use crate::hittable::*;
use crate::materials::{MaterialId, MaterialTable};
use crate::math::*;

pub use crate::accelerator::{Accelerator, AcceleratorType};
pub use crate::geometry::Instance;

pub struct World {
    pub accelerator: Accelerator,
    pub lights: Vec<usize>,
    pub materials: MaterialTable,
    pub background: MaterialId,
}

impl World {
    pub fn pick_random_light(&self, s: &mut Box<dyn Sampler>) -> Option<&Instance> {
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
            Some(self.accelerator.get_primitive(self.lights[idx]))
        }
    }

    pub fn get_primitive(&self, index: usize) -> &Instance {
        self.accelerator.get_primitive(index)
    }

    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        self.accelerator.hit(r, t0, t1)
    }
}

// impl HasBoundingBox for World {
//     fn bounding_box(&self) -> AABB {
//         self.accelerator.bounding_box()
//     }
// }
