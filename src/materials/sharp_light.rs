use crate::hittable::HitRecord;
use crate::material::Material;
use crate::math::*;

#[derive(Clone, Debug)]
pub struct SharpLight {
    // pub color: Box<dyn SpectralPowerDistribution>,
    pub color: CDF,
    pub sharpness: f32,
    pub sidedness: Sidedness,
}

impl SharpLight {
    pub fn new(color: CDF, sharpness: f32, sidedness: Sidedness) -> SharpLight {
        SharpLight {
            color,
            sharpness: 1.0 + sharpness,
            sidedness,
        }
    }
    pub const NAME: &'static str = "SharpLight";
}

fn evaluate(vec: Vec3, sharpness: f32) -> f32 {
    let cos_phi = vec.z().clamp(0.0, 1.0);
    // assert!(cos_phi <= 1.0 && cos_phi >= 0.0);
    let cos_phi2 = cos_phi * cos_phi;
    let sin_phi2 = 1.0 - cos_phi2;
    // let sin_phi = sin_phi2.sqrt();
    let sharpness2 = sharpness * sharpness;

    let common1 = sharpness2 * (2.0 * cos_phi2 - 1.0);
    // assert!(
    //     1.0 - sharpness2 * sin_phi2 >= 0.0,
    //     "{}*{} = {}",
    //     sharpness2,
    //     sin_phi2,
    //     sharpness2 * sin_phi2
    // );
    let common2 = 2.0 * sharpness * cos_phi * (1.0 - sharpness2 * sin_phi2).max(0.0).sqrt();
    assert!(1.0 + common1 + common2 >= 0.0);
    assert!(1.0 + common1 - common2 >= 0.0);
    let dist_top = (1.0 + common1 + common2).sqrt();
    let dist_bottom = (1.0 + common1 - common2).sqrt();
    assert!(
        dist_bottom.is_finite() && dist_top.is_finite(),
        "{} {}",
        dist_bottom,
        dist_top
    );
    (dist_top - dist_bottom) / (2.0 * PI)
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

        // let mut local_wo =
        // (random_cosine_direction(scatter_sample) + self.sharpness * Vec3::Z).normalized();
        let mut non_normalized_local_wo = if self.sharpness == 1.0 {
            random_cosine_direction(scatter_sample)
        } else {
            random_on_unit_sphere(scatter_sample) + Vec3::Z * self.sharpness
        };
        // let mut local_wo = Vec3::Z;
        let fac = evaluate(non_normalized_local_wo.normalized(), self.sharpness);

        if swap {
            non_normalized_local_wo = -non_normalized_local_wo;
        }

        assert!(fac.is_finite(), "{:?}, {:?}", self, non_normalized_local_wo);
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame
            .to_world(&non_normalized_local_wo.normalized())
            .normalized();
        // let directional_pdf = local_wo.z().abs() / PI;
        // debug_assert!(directional_pdf > 0.0, "{:?} {:?}", local_wo, object_wo);
        let (sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        // fac both affects the power of the emitted light and the pdf.
        Some((
            Ray::new(point, object_wo),
            sw.with_energy(sw.energy * fac),
            PDF::from(fac),
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
        // wi is in local space, and is normalized
        // lets check if it could have been constructed by sample_emission.
        let cosine = wi.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            let min_z = (1.0 - self.sharpness.powi(2).recip()).sqrt();
            if cosine > min_z {
                // could have been generated
                let fac = evaluate(wi, self.sharpness);
                SingleEnergy::new(fac * self.color.evaluate_power(hit.lambda))
            } else {
                SingleEnergy::ZERO
            }
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
            let min_z = (1.0 - self.sharpness.powi(2).recip()).sqrt();
            if cosine > min_z {
                // could have been generated
                let pdf = evaluate(wo, self.sharpness);
                pdf.into()
            } else {
                0.0.into()
            }
        } else {
            0.0.into()
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::curves;
//     #[test]
//     fn test_integral() {
//         let light = SharpLight::new(curves::void(), 4.0, Sidedness::Forward);
//         for _ in 0..10000 {
//             let generated = light.sample_emission(
//                 Point3::ORIGIN,
//                 Vec3::Z,
//                 curves::EXTENDED_VISIBLE_RANGE,
//                 Sample2D::new_random_sample(),
//                 Sample1D::new_random_sample(),
//             );
//         }
//     }
// }
