use math::Sidedness;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct SharpLight {
    pub bounce_color: Curve,
    pub emit_color: CurveWithCDF,
    pub sharpness: f32,
    pub sidedness: Sidedness,
}

impl SharpLight {
    pub fn new(
        bounce_color: Curve,
        emit_color: CurveWithCDF,
        sharpness: f32,
        sidedness: Sidedness,
    ) -> SharpLight {
        SharpLight {
            bounce_color,
            emit_color,
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

impl Material<f32, f32> for SharpLight {
    #[allow(unused_variables)]
    fn bsdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        (0.0, 0.0.into())
    }

    #[allow(unused_variables)]
    fn generate_and_evaluate(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        (0.0, None, 0.0.into())
    }

    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        mut scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(
        Ray,
        SingleWavelength,
        PDF<f32, SolidAngle>,
        PDF<f32, Uniform01>,
    )> {
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
        let (sw, wavelength_pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        // fac both affects the power of the emitted light and the pdf.
        Some((
            Ray::new(point, object_wo),
            sw.replace_energy(sw.energy * fac),
            PDF::from(fac),
            wavelength_pdf,
        ))
    }
    fn emission(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
    ) -> f32 {
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
                fac * self.emit_color.evaluate_power(lambda)
            } else {
                f32::ZERO
            }
        } else {
            f32::ZERO
        }
    }

    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wo: Vec3,
    ) -> PDF<f32, SolidAngle> {
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

    fn sample_emission_spectra(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF<f32, Uniform01>)> {
        let (sw, pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((sw.lambda, pdf))
    }
}

#[cfg(test)]
mod test {
    use minifb::WindowOptions;

    use crate::{curves, tonemap::Clamp};

    use super::*;

    #[test]
    fn test_sampling_direction() {
        let light = SharpLight::new(
            curves::void(),
            curves::blackbody_curve(5000.0, 1.0).to_cdf(EXTENDED_VISIBLE_RANGE, 100),
            1.0,
            Sidedness::Forward,
        );

        let (width, height) = (500, 500);
        let mut film = Film::new(width, height, XYZColor::BLACK);
        for _ in 0..10000 {
            let out = light.sample_emission(
                Point3::ORIGIN,
                Vec3::Z,
                BOUNDED_VISIBLE_RANGE,
                Sample2D::new_random_sample(),
                Sample1D::new_random_sample(),
            );
            if let Some((ray, packet, solid_angle_pdf, wavelength_pdf)) = out {
                let uv = direction_to_uv(ray.direction);
                let p = (
                    (uv.0 * width as f32) as usize,
                    (uv.1 * height as f32) as usize,
                );

                let jacobian = (uv.1 * PI).sin().abs() * 2.0 * PI * PI;
                let pixel = film.at(p.0, p.1);
                film.write_at(
                    p.0,
                    p.1,
                    pixel
                        + SingleWavelength::new(
                            packet.lambda,
                            packet.energy / (*wavelength_pdf * *solid_angle_pdf / jacobian),
                        )
                        .into(),
                );
            }
        }
        let mut tonemapper = Clamp::new(0.0, true, true);
        let mut total_samples = 10000;
        let samples_per_iteration = 1000;
        window_loop(
            width,
            height,
            144,
            WindowOptions::default(),
            true,
            |_, mut window_buffer, width, height| {
                for _ in 0..samples_per_iteration {
                    let out = light.sample_emission(
                        Point3::ORIGIN,
                        Vec3::Z,
                        EXTENDED_VISIBLE_RANGE,
                        Sample2D::new_random_sample(),
                        Sample1D::new_random_sample(),
                    );
                    if let Some((ray, packet, solid_angle_pdf, wavelength_pdf)) = out {
                        // let sample = Sample2D::new_random_sample();

                        let uv = direction_to_uv(ray.direction);
                        let p = (
                            (uv.0 * width as f32) as usize,
                            (uv.1 * height as f32) as usize,
                        );

                        let jacobian = (uv.1 * PI).sin().abs() * 2.0 * PI * PI;
                        let pixel = film.at(p.0, p.1);
                        film.write_at(
                            p.0,
                            p.1,
                            pixel
                                + SingleWavelength::new(
                                    packet.lambda,
                                    packet.energy / (*wavelength_pdf * *solid_angle_pdf / jacobian),
                                )
                                .into(),
                        );
                        total_samples += 1;
                    }
                }

                let factor = 1.0 / ((total_samples as f32).sqrt() + 1.0);
                update_window_buffer(
                    &mut window_buffer,
                    &film,
                    &mut tonemapper,
                    Converter::sRGB,
                    factor,
                );
            },
        );
    }
}
