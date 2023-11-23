use math::Sidedness;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct SharpLight {
    // is actually just a light with a cosine power weighted light emission distribution and pdf
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
        // sharpness can be almost any f32 value, will have `.abs()` and `+ 1.0` performed on it.
        SharpLight {
            bounce_color,
            emit_color,
            sharpness: 1.0 + sharpness.abs(),
            sidedness,
        }
    }
    pub const NAME: &'static str = "SharpLight";
}

fn random_weighted_cosine(sample: Sample2D, power: f32) -> Vec3 {
    let theta = sample.x.powf((1.0 + power).recip()).acos();
    let phi = TAU * sample.y;

    let (phi_sin, phi_cos) = phi.sin_cos();
    let (theta_sin, theta_cos) = theta.sin_cos();

    Vec3::new(phi_cos * theta_sin, phi_sin * theta_sin, theta_cos)
}

impl Material<f32, f32> for SharpLight {
    fn bsdf(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        // copy from lambertian
        if wo.z() * wi.z() > 0.0 {
            (
                self.bounce_color.evaluate_clamped(lambda) / PI,
                (wo.z().abs() / PI).into(),
            )
        } else {
            (0.0.into(), 0.0.into())
        }
    }
    fn generate(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        // bounce, copy from lambertian

        let d = random_cosine_direction(s) * wi.z().signum();
        Some(d)
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

        let mut local_wo = if self.sharpness == 1.0 {
            random_cosine_direction(scatter_sample)
        } else {
            random_weighted_cosine(scatter_sample, self.sharpness)
        };
        // let mut local_wo = Vec3::Z;
        let pdf = Self::evaluate_inner(local_wo, self.sharpness);
        assert!(
            pdf != 0.0,
            "{} {} {:?} {} {}",
            pdf,
            self.sharpness,
            local_wo,
            scatter_sample.x,
            scatter_sample.y
        );

        if swap {
            local_wo = -local_wo;
        }

        assert!(pdf.is_finite(), "{:?}, {:?}", self, local_wo);
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame.to_world(&local_wo.normalized()).normalized();
        // let directional_pdf = local_wo.z().abs() / PI;
        // debug_assert!(directional_pdf > 0.0, "{:?} {:?}", local_wo, object_wo);
        let (sw, wavelength_pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        // fac both affects the power of the emitted light and the pdf.
        Some((
            Ray::new(point, object_wo),
            sw.replace_energy(sw.energy * pdf),
            PDF::from(pdf),
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
            self.emit_color.evaluate_power(lambda) * Self::evaluate_inner(wi, self.sharpness)
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
            Self::evaluate_inner(wo, self.sharpness).into()
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

    fn generate_and_evaluate(
        &self,
        lambda: f32,
        _: (f32, f32),
        _: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        let d = random_cosine_direction(s) * wi.z().signum();

        (
            self.bounce_color.evaluate_clamped(lambda) / PI,
            Some(d),
            (d.z().abs() / PI).into(),
        )
    }
}

impl SharpLight {
    pub fn evaluate_inner(vec: Vec3, sharpness: f32) -> f32 {
        (sharpness + 1.0) * vec.z().abs().powf(sharpness) / 2.0 / PI
    }
    pub fn evaluate(&self, vec: Vec3) -> f32 {
        Self::evaluate_inner(vec, self.sharpness)
    }
}

#[cfg(test)]
mod test {

    use minifb::WindowOptions;

    use crate::{curves, tonemap::Clamp};

    use super::*;

    #[test]
    fn test_evaluate() {
        let non_normalized_local_wo = Vec3::new(0.8398655, -0.2606247, 1.623869);
        let sharpness = 2.1;
        let e = SharpLight::evaluate_inner(non_normalized_local_wo.normalized(), sharpness);
        println!("{}", e);
    }

    #[test]
    fn test_pdf_integrates_to_one() {
        for _ in 0..10 {
            let mut sharpness = rand::random::<f32>() * 100.0;
            if sharpness < 1.0 {
                sharpness = 0.0;
            }

            println!("testing with sharpness {}", sharpness);

            let light = SharpLight::new(
                curves::void(),
                curves::blackbody_curve(5000.0, 1.0).to_cdf(EXTENDED_VISIBLE_RANGE, 100),
                sharpness,
                Sidedness::Forward,
            );

            // three ways, sampling randomly, importance sampling, and using the quadrature rule

            // quadrature rule first
            {
                let mut sum = 0.0;
                let mut num_samples = 0;

                // need to actually multiply in the higher dimensional analogue of
                // the delta_x factor for numerical integration.
                let theta_n = 100;
                let phi_n = 1000;
                for theta_i in 0..theta_n {
                    for phi_i in 1..=phi_n {
                        let theta = TAU * theta_i as f32 / theta_n as f32;
                        let x = phi_i as f32 / phi_n as f32;
                        let phi = PI * x;

                        let (phi_sin, z) = phi.sin_cos();
                        let (theta_sin, theta_cos) = theta.sin_cos();
                        let jacobian = phi_sin.abs() * PI * PI;
                        let (x, y) = (theta_cos * phi_sin, theta_sin * phi_sin);

                        let v = Vec3::new(x, y, z);

                        let pdf = light.evaluate(v);
                        sum += pdf * jacobian;
                        num_samples += 1;
                    }
                }
                println!("{:?}", sum / num_samples as f32);
            }
            {
                // then uniform sampling (technically uniform over angles, so multiply pdf by proper jacobian)
                let mut sum = 0.0;
                let mut num_samples = 0;

                for _ in 0..100000 {
                    let theta = rand::random::<f32>() * TAU;
                    let x = rand::random::<f32>();
                    let phi = x * PI;

                    let (phi_sin, z) = phi.sin_cos();
                    let (theta_sin, theta_cos) = theta.sin_cos();
                    let (x, y) = (theta_cos * phi_sin, theta_sin * phi_sin);

                    let jacobian = phi_sin.abs() * PI * PI;
                    let v = Vec3::new(x, y, z);

                    let pdf = light.evaluate(v);
                    sum += pdf * jacobian;
                    num_samples += 1;
                }
                // with sharpness at 0.0, this evaluates to 1/pi.
                println!("{:?}", sum / num_samples as f32);
            }
        }
    }

    #[test]
    fn test_sampling_direction() {
        let light = SharpLight::new(
            curves::void(),
            curves::blackbody_curve(5000.0, 1.0).to_cdf(EXTENDED_VISIBLE_RANGE, 100),
            100.0,
            Sidedness::Forward,
        );
        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1usize)
        //     .build_global()
        //     .unwrap();

        let (width, height) = (500, 500);
        let mut film = Vec2D::new(width, height, XYZColor::BLACK);

        let mut sum = XYZColor::BLACK;
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
                let new_color = SingleWavelength::new(
                    packet.lambda,
                    packet.energy / (*wavelength_pdf * *solid_angle_pdf / jacobian),
                )
                .into();
                sum += new_color;
                film.write_at(p.0, p.1, pixel + new_color);
            }
        }
        println!("{:?}", sum / 10000.0);

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
                        let sw = SingleWavelength::new(
                            packet.lambda,
                            packet.energy / (*wavelength_pdf * *solid_angle_pdf / jacobian),
                        );
                        film.write_at(p.0, p.1, pixel + sw.into());
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
