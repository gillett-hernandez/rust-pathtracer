use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;

pub struct DiffuseLight {
    // pub color: Box<dyn SpectralPowerDistribution>,
    pub color: SPD,
    pub sidedness: Sidedness,
}

impl DiffuseLight {
    pub fn new(color: SPD, sidedness: Sidedness) -> DiffuseLight {
        DiffuseLight { color, sidedness }
    }
}

impl PDF for DiffuseLight {
    fn value(&self, _hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> f32 {
        0.0
    }
    fn generate(&self, _hit: &HitRecord, _s: &mut Box<dyn Sampler>, _wi: Vec3) -> Option<Vec3> {
        None
    }
}

impl BRDF for DiffuseLight {
    fn f(&self, _hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> SingleEnergy {
        SingleEnergy::ZERO
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
        if (wi.z() > 0.0 && self.sidedness == Sidedness::Forward)
            || (wi.z() < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            SingleEnergy::new(self.color.evaluate_power(hit.lambda) / PI)
        } else {
            SingleEnergy::ZERO
        }
    }
}

impl Material for DiffuseLight {}
