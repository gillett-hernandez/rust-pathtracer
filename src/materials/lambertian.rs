use crate::hittable::HitRecord;
use crate::materials::Material;
use crate::math::*;
use crate::texture::TexStack;

#[derive(Clone)]
pub struct Lambertian {
    pub color: TexStack,
}

impl Lambertian {
    pub fn new(color: TexStack) -> Lambertian {
        Lambertian { color }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for Lambertian {
    fn scatter_pdf(&self, _hit: &HitRecord, wi: Vec3, wo: Vec3) -> PDF {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (cosine / PI).into()
        } else {
            0.0.into()
        }
    }
    fn generate(&self, _hit: &HitRecord, s: Sample2D, _wi: Vec3) -> Option<Vec3> {
        Some(random_cosine_direction(s))
    }
    // don't implement sample_emission, since the default implementation is what we want.
    // though perhaps it would be a good idea to panic if a the integrator tries to sample the emission of a lambertian

    // implement f

    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> SingleEnergy {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            SingleEnergy::new(self.color.eval_at(hit.lambda, hit.uv).min(1.0) / PI)
        } else {
            0.0.into()
        }
    }
    // fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
    //     SingleEnergy::ZERO
    // }
}
