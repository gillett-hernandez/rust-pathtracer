use crate::hittable::HitRecord;
use crate::material::Material;
use crate::math::*;

#[derive(Clone, Debug)]
pub struct DiffuseLight {
    // pub color: Box<dyn SpectralPowerDistribution>,
    pub color: CDF,
    pub sidedness: Sidedness,
}

impl DiffuseLight {
    pub fn new(color: CDF, sidedness: Sidedness) -> DiffuseLight {
        DiffuseLight { color, sidedness }
    }
}

impl Material for DiffuseLight {
    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        mut scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength, PDF, PDF)> {
        // wo localized to point and normal
        let mut swap = false;
        if self.sidedness == Sidedness::Reverse {
            swap = true;
        } else if self.sidedness == Sidedness::Dual {
            if scatter_sample.x < 0.5 {
                swap = true;
                scatter_sample.x *= 2.0;
            } else {
                scatter_sample.x = (1.0 - scatter_sample.x) * 2.0;
            }
        }
        let mut local_wo = random_cosine_direction(scatter_sample) + 0.0001 * Vec3::Z;

        if swap {
            local_wo = -local_wo;
        }
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame.to_world(&local_wo).normalized();
        let directional_pdf = local_wo.z().abs() / PI;
        debug_assert!(directional_pdf > 0.0, "{:?} {:?}", local_wo, object_wo);
        let (sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((
            Ray::new(point, object_wo),
            sw.with_energy(sw.energy / PI),
            PDF::from(directional_pdf),
            pdf,
        ))
    }

    fn sample_emission_spectra(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF)> {
        let (sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((sw.lambda, pdf))
    }

    fn emission(&self, hit: &HitRecord, wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
        let cosine = wi.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            // if wi.z() > 0.0 {
            SingleEnergy::new(self.color.evaluate_power(hit.lambda) / PI)
        } else {
            SingleEnergy::ZERO
        }
    }

    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(&self, _hit: &HitRecord, wo: Vec3) -> PDF {
        let cosine = wo.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            (cosine / PI).into()
        } else {
            0.0.into()
        }
    }
}
