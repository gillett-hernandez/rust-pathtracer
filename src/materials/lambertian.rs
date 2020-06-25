use crate::hittable::HitRecord;
use crate::material::Material;
use crate::math::*;
pub struct Lambertian {
    pub color: SPD,
}

impl Lambertian {
    pub fn new(color: SPD) -> Lambertian {
        Lambertian { color }
    }
}

impl Material for Lambertian {
    fn value(&self, _hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            cosine / PI
        } else {
            0.0
        }
    }
    fn generate(&self, _hit: &HitRecord, s: Sample2D, _wi: Vec3) -> Option<Vec3> {
        Some(random_cosine_direction(s))
    }
    // don't implement sample_emission, since the default implementation is what we want.
    // though perhaps it would be a good idea to panic if a the integrator tries to sample the emission of a lambertian

    // implement f

    fn f(&self, hit: &HitRecord, _wi: Vec3, _wo: Vec3) -> SingleEnergy {
        SingleEnergy::new(self.color.evaluate(hit.lambda) / PI)
    }
    // fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
    //     SingleEnergy::ZERO
    // }
}
