use crate::hittable::HitRecord;
use crate::material::Material;
use crate::math::*;
use crate::NORMAL_OFFSET;

#[derive(Clone, Debug)]
pub struct ParallelLight {
    // pub color: Box<dyn SpectralPowerDistribution>,
    pub color: SPD,
    pub sidedness: Sidedness,
}

impl ParallelLight {
    pub fn new(color: SPD, sidedness: Sidedness) -> ParallelLight {
        ParallelLight { color, sidedness }
    }
}

impl Material for ParallelLight {
    // don't implement the other functions, since the fallback default implementation does the exact same thing

    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        mut scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength, PDF)> {
        // wo localized to point and normal
        let mut swap = false;
        if self.sidedness == Sidedness::Reverse {
            swap = true;
        }

        if self.sidedness == Sidedness::Dual {
            if scatter_sample.x < 0.5 {
                swap = true;
                scatter_sample.x *= 2.0;
            } else {
                scatter_sample.x = (1.0 - scatter_sample.x) * 2.0;
            }
        }
        // let mut local_wo = (random_cosine_direction(scatter_sample) + 10.0 * Vec3::Z).normalized();
        let mut local_wo = Vec3::Z;

        if swap {
            local_wo = -local_wo;
        }
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame.to_world(&local_wo).normalized();
        let (sw, _pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((
            Ray::new(point + object_wo * NORMAL_OFFSET, object_wo),
            // sw.with_energy(sw.energy / PI),
            sw,
            // PDF::from(local_wo.z().abs() / PI),
            PDF::from(0.0),
            // PDF::from(local_wo.z().abs() * pdf.0 / PI),
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
        if (wi.z() > 0.0 && self.sidedness == Sidedness::Forward)
            || (wi.z() < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            if wi.z().abs() > 0.9 {
                SingleEnergy::new(self.color.evaluate_power(hit.lambda) / PI)
            } else {
                SingleEnergy::ZERO
            }
        } else {
            SingleEnergy::ZERO
        }
    }
}
