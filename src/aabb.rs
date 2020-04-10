use crate::math::*;

pub trait HasBoundingBox {
    fn bounding_box(&self) -> AABB;
}
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> AABB {
        AABB { min, max }
    }
    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> bool {
        let mut tmin: f32 = t0;
        let mut tmax: f32 = t1;
        let mut t0: f32;
        let mut t1: f32;
        t0 = ((self.min.x - r.origin.x) / r.direction.x)
            .min((self.max.x - r.origin.x) / r.direction.x);
        t1 = ((self.min.x - r.origin.x) / r.direction.x)
            .max((self.max.x - r.origin.x) / r.direction.x);
        if tmax <= tmin {
            return false;
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);

        t0 = ((self.min.y - r.origin.y) / r.direction.y)
            .min((self.max.y - r.origin.y) / r.direction.y);
        t1 = ((self.min.y - r.origin.y) / r.direction.y)
            .max((self.max.y - r.origin.y) / r.direction.y);
        if tmax <= tmin {
            return false;
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);

        t0 = ((self.min.z - r.origin.z) / r.direction.z)
            .min((self.max.z - r.origin.z) / r.direction.z);
        t1 = ((self.min.z - r.origin.z) / r.direction.z)
            .max((self.max.z - r.origin.z) / r.direction.z);
        if tmax <= tmin {
            return false;
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);

        true
    }
}
