use crate::hittable::HitRecord;
use crate::material::Material;
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

impl Material for DiffuseLight {
    // don't implement the other functions, since the fallback default implementation does the exact same thing

    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength)> {
        // wo localized to point and normal
        let local_wo = random_cosine_direction(scatter_sample);

        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame.to_world(&local_wo).normalized();
        let sw = self.color.sample_power(wavelength_range, wavelength_sample);
        Some((Ray::new(point, object_wo), sw))
    }

    fn emission(&self, hit: &HitRecord, wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
        if (wi.z() > 0.0 && self.sidedness == Sidedness::Forward)
            || (wi.z() < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            SingleEnergy::new(self.color.evaluate_power(hit.lambda))
        } else {
            SingleEnergy::ZERO
        }
    }
}
