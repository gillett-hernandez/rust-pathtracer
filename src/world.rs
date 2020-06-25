use packed_simd::f32x4;

use crate::hittable::*;
use crate::materials::{MaterialId, MaterialTable};
use crate::math::*;

pub use crate::accelerator::{Accelerator, AcceleratorType};
pub use crate::geometry::Instance;

pub struct EnvironmentMap {
    pub color: SPD,
}

impl EnvironmentMap {
    pub const fn new(color: SPD) -> Self {
        EnvironmentMap { color }
    }
    pub fn sample_spd(
        &self,
        uv: (f32, f32),
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> Option<SingleWavelength> {
        Some(self.color.sample_power(wavelength_range, sample))
    }

    pub fn emission(&self, _uv: (f32, f32), lambda: f32) -> SingleEnergy {
        SingleEnergy::new(self.color.evaluate_power(lambda))
    }

    pub fn sample_emission(
        &self,
        world_radius: f32,
        uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength) {
        let phi = PI * (2.0 * uv.0 - 1.0);
        let (y, x) = phi.sin_cos();
        let z = (PI * uv.1).cos();
        let point = Point3::from_raw((f32x4::new(x, y, z, 0.0) * world_radius).replace(3, 1.0));
        let direction = Vec3::new(-x, -y, -z);
        let sw = self.color.sample_power(wavelength_range, wavelength_sample);

        // force overwrite point and direction for testing purposes
        let point = Point3::new(0.0, 0.0, world_radius);
        let direction = -Vec3::Z;
        (Ray::new(point, direction), sw)
    }
}

pub struct World {
    pub accelerator: Accelerator,
    pub lights: Vec<usize>,
    pub materials: MaterialTable,
    pub environment: EnvironmentMap,
}

impl World {
    pub fn pick_random_light(&self, s: Sample1D) -> Option<&Instance> {
        let length = self.lights.len();
        if length == 0 {
            None
        } else {
            let x = s.x;
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

impl HasBoundingBox for World {
    fn bounding_box(&self) -> AABB {
        self.accelerator.bounding_box()
    }
}
