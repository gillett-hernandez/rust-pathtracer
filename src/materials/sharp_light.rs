use crate::hittable::HitRecord;
use crate::material::Material;
use crate::math::*;

#[derive(Clone, Debug)]
pub struct SharpLight {
    // pub color: Box<dyn SpectralPowerDistribution>,
    pub color: SPD,
    pub sharpness: f32,
    pub sidedness: Sidedness,
}

impl SharpLight {
    pub fn new(color: SPD, sharpness: f32, sidedness: Sidedness) -> SharpLight {
        SharpLight {
            color,
            sharpness: 1.0 + sharpness,
            sidedness,
        }
    }
}

impl Material for SharpLight {
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
        } else if self.sidedness == Sidedness::Dual {
            if scatter_sample.x < 0.5 {
                swap = true;
                scatter_sample.x *= 2.0;
            } else {
                scatter_sample.x = (1.0 - scatter_sample.x) * 2.0;
            }
        }

        // let mut local_wo =
        // (random_cosine_direction(scatter_sample) + self.sharpness * Vec3::Z).normalized();
        let mut non_normalized_local_wo = if self.sharpness == 1.0 {
            random_cosine_direction(scatter_sample)
        } else {
            random_on_unit_sphere(scatter_sample) + Vec3::Z * self.sharpness
        };
        // let mut local_wo = Vec3::Z;

        if swap {
            non_normalized_local_wo = -non_normalized_local_wo;
        }
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame
            .to_world(&non_normalized_local_wo.normalized())
            .normalized();
        // let directional_pdf = local_wo.z().abs() / PI;
        // debug_assert!(directional_pdf > 0.0, "{:?} {:?}", local_wo, object_wo);
        let (sw, _pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((
            Ray::new(point, object_wo),
            sw.with_energy(sw.energy),
            PDF::from(non_normalized_local_wo.z().abs() / PI),
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
        // wi is in local space, and is normalized
        // lets check if it could have been constructed by sample_emission.

        let min_z = (1.0 - self.sharpness.powi(2).recip()).sqrt();
        if wi.z() >= min_z {
            // could have been generated
            SingleEnergy::new(self.color.evaluate_power(hit.lambda) / PI)
        } else {
            SingleEnergy::ZERO
        }
    }
}
