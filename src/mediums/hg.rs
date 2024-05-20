use crate::prelude::*;

use super::Medium;

pub fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    debug_assert!(
        denom > 0.0,
        "1.0 + {:?} + {:?} == {:?}",
        g * g,
        2.0 * g * cos_theta,
        denom
    );
    (1.0 - g * g) / (denom * denom.sqrt() * 2.0 * std::f32::consts::TAU)
}
#[derive(Clone)]
pub struct HenyeyGreensteinHomogeneous {
    // domain: visible range
    // range: 0..2
    // actual g = g - 1
    pub g: Curve,
    pub sigma_a: Curve, // absorption attenuation
    pub sigma_s: Curve, // scattering attenuation
}

impl HenyeyGreensteinHomogeneous {
    pub fn new(g: Curve, sigma_a: Curve, sigma_s: Curve) -> Self {
        Self {
            g,
            sigma_a,
            sigma_s,
        }
    }

    pub fn sigma_t(&self, lambda: f32) -> f32 {
        self.sigma_a.evaluate(lambda) + self.sigma_s.evaluate(lambda)
    }
}

impl Medium<f32, f32> for HenyeyGreensteinHomogeneous {
    fn p(&self, lambda: f32, _uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32 {
        let cos_theta = wi * wo;

        let g = self.g.evaluate_power(lambda) + 0.001 - 1.0;
        let phase = phase_hg(cos_theta, g);
        let sigma_s = self.sigma_s.evaluate_power(lambda);

        let v = sigma_s * phase;
        debug_assert!(
            v.is_finite(),
            "{:?}, {:?}, {:?}, {:?}",
            sigma_s,
            phase,
            g,
            cos_theta
        );
        v
    }
    fn sample_p(
        &self,
        lambda: f32,
        _uvw: (f32, f32, f32),
        wi: Vec3,
        s: Sample2D,
    ) -> (Vec3, PDF<f32, SolidAngle>) {
        let g = self.g.evaluate_power(lambda) + 0.001 - 1.0;
        let cos_theta = if g.abs() < 0.001 {
            1.0 - 2.0 * s.x
        } else {
            let sqr = (1.0 - g * g) / (1.0 + g - 2.0 * g * s.x);
            -(1.0 + g * g - sqr * sqr) / (2.0 * g)
        };

        let sin_theta = (0.0f32).max(1.0 - cos_theta * cos_theta).sqrt();
        let phi = std::f32::consts::TAU * s.y;
        let frame = TangentFrame::from_normal(wi);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let wo = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let pdf = phase_hg(cos_theta, g);
        // let sigma_s = self.sigma_s.evaluate_power(lambda);
        // let v = sigma_s * pdf;
        debug_assert!(pdf.is_finite(), "{:?}, {:?}, {:?}", pdf, g, cos_theta);

        (frame.to_world(&wo), pdf.into())
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        let sigma_s = self.sigma_s.evaluate(lambda);
        let dist = -(1.0 - s.x).ln() / sigma_s;
        let t = dist.min(ray.tmax);
        let sampled_medium = t < ray.tmax;

        let point = ray.point_at_parameter(t);
        let tr = self.tr(lambda, ray.origin, point);

        if sampled_medium {
            (point, tr, true)
        } else {
            (point, tr, false)
            // (point, 1.0, false)
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma_t = self.sigma_t(lambda);
        (-sigma_t * (p1 - p0).norm()).exp()
    }
}

#[cfg(feature = "preview")]
#[cfg(test)]
mod test {
    use minifb::WindowOptions;
    use rayon::prelude::*;

    use crate::tonemap::Clamp;

    use super::*;

    #[allow(dead_code)]
    enum TestMode {
        ViewPhase,
        SamplePhase,
    }
    #[test]
    fn test_phase() {
        let width = 500usize;
        let height = 500usize;
        let bounds = BOUNDED_VISIBLE_RANGE;

        let mut film = Vec2D::new(width, height, XYZColor::BLACK);

        // we want the calculated sigma to be on the order of 5.1 * 10.0f32.powi(-31),
        // we also want to be able to calculate that medium.tr(532.0, Point3::ORIGIN, Point3::new(0.0, 0.0, 1.0))
        // should be about 10^-5

        #[rustfmt::skip]
        let medium = HenyeyGreensteinHomogeneous::new(
            cie_e(1.5), // forward scatter
            cie_e(0.1), // constant sigma_s for wavelength
            cie_e(0.34), // constant sigma_s for wavelength
        );

        let mode = TestMode::ViewPhase;
        // let mode = TestMode::SamplePhase;
        // let wi = Vec3::new(0.0, 1.0, 1.0).normalized();
        let wi = Vec3::Z;
        let mut total_samples = 0;

        let (samples_per_iteration, exposure): (usize, f32) = match mode {
            TestMode::ViewPhase => (10, 1.0),
            TestMode::SamplePhase => (1000, 1.0),
        };

        let test_wo = -wi;
        // let test_lambda = bounds.sample(random());
        let test_lambda = 532.0f32;

        let sigma_t = medium.sigma_t(test_lambda);
        println!("sigma_t = {:?}", sigma_t);

        let test_phase = medium.p(test_lambda, (0.0, 0.0, 0.0), wi, test_wo);
        println!("test phase = {:?}", test_phase);

        let test_transmittance = medium.tr(test_lambda, Point3::new(0.0, 0.0, 1.0), Point3::ORIGIN);

        println!("{:?}", test_transmittance);
        let test_transmittance =
            medium.tr(test_lambda, Point3::new(0.0, 0.0, 10.0), Point3::ORIGIN);
        println!("{:?}", test_transmittance);

        let mut tonemapper = Clamp::new(exposure, true, true);
        window_loop(
            width,
            height,
            144,
            WindowOptions::default(),
            true,
            |_, mut window_buffer, width, height| {
                let factor = match mode {
                    TestMode::ViewPhase => {
                        film.buffer
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(idx, pixel)| {
                                let mut local_sum = XYZColor::BLACK;
                                let p = (idx % width, idx / width);
                                for _ in 0..samples_per_iteration {
                                    let uv = (
                                        (p.0 as f32 + debug_random()) / width as f32,
                                        (p.1 as f32 + debug_random()) / height as f32,
                                    );
                                    let lambda = bounds.sample(debug_random());
                                    let wo = uv_to_direction(uv);
                                    let phase = medium.p(lambda, (0.0, 0.0, 0.0), wi, wo);

                                    local_sum +=
                                        XYZColor::from(SingleWavelength::new(lambda, phase.into()));
                                }
                                *pixel += local_sum / samples_per_iteration as f32;
                            });
                        total_samples += 1;

                        1.0 / (total_samples as f32 + 1.0)
                    }
                    TestMode::SamplePhase => {
                        for i in 0..samples_per_iteration {
                            let sample = Sample2D::new_random_sample();
                            let lambda = bounds.sample(debug_random());
                            let (wo, pdf) = medium.sample_p(lambda, (0.0, 0.0, 0.0), wi, sample);

                            let uv = direction_to_uv(wo);
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
                                    + SingleWavelength::new(lambda, (*pdf / jacobian).into())
                                        .into(),
                            );
                            if i == 0 {
                                println!("{:?}, {:?}", wo, pdf);
                            }
                        }
                        total_samples += samples_per_iteration;

                        1.0 / ((total_samples as f32).sqrt() + 1.0)
                    }
                };
                update_window_buffer(&mut window_buffer, &film, &mut tonemapper, factor);
            },
        );
    }
}
