use crate::prelude::*;

use super::Medium;

#[derive(Clone)]
pub struct Rayleigh {
    pub corrective_factor: f32,
    pub ior: Curve,
}

// FIXME: correct scale of certain constants such that a reasonable sigma and phase are outputted, i.e.
// such that they don't require a massive scale factor to cause things to actually render correctly.

impl Rayleigh {
    pub fn new(corrective_factor: f32, ior: Curve) -> Self {
        // let sixth_root = particles_per_m_cubed.cbrt().sqrt();
        let self_ = Self {
            corrective_factor,
            ior,
        };
        #[cfg(test)]
        {}
        self_
    }
    fn ior_factor(&self, lambda: f32) -> f32 {
        let n = self.ior.evaluate(lambda);
        let n_2 = n * n;
        let numerator = n_2 - 1.0;
        let denominator = n_2 + 2.0;
        (numerator / denominator).powi(2)
    }
    pub fn sigma_s(&self, lambda: f32) -> f32 {
        let ior_factor = self.ior_factor(lambda);

        // convert to micrometers before applying rayleigh lambda stuff, to avoid floating point issues.
        let lambda_factor = (lambda / 1000.0).recip().powi(4);

        ior_factor * self.corrective_factor * lambda_factor
    }
    // TODO: figure out if rayleigh mediums have a sigma_a at all, or if sigma_t == sigma_s
}

impl Medium<f32, f32> for Rayleigh {
    fn p(&self, lambda: f32, _uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32 {
        // I = I_0 * (1+cos^2(theta)) / 2R^2 * (2pi/lambda)^4 * ((n^2-1) / (n^2 + 2))^2 * (d/2)^6
        // where:
        //      R = distance to observer
        //      theta = scattering angle
        //      n = index of refraction
        //      d = averaged particle diameter

        let cos_squared = (wi * wo).powi(2);

        let ior_factor = self.ior_factor(lambda);

        let lambda_factor = (lambda / 1000.0).recip().powi(4);
        (1.0 + cos_squared) * 3.0 * ior_factor * lambda_factor * self.corrective_factor / 8.0
    }
    fn sample_p(
        &self,
        _lambda: f32,
        _uvw: (f32, f32, f32),
        wi: Vec3,
        sample: Sample2D,
    ) -> (Vec3, PDF<f32, SolidAngle>) {
        // let vec = random_on_unit_sphere(sample);
        // let pdf = (4.0 * PI).recip();

        // inverse transform sample to generate a cos_theta

        let (x, flipped) = Sample1D::new(sample.x).choose(0.5, true, false);
        let z = 2.0 * (2.0 * x.x - 1.0); // 4x jacobian here, i.e. dz/dx = 4x
        let right = (z * z + 1.0).sqrt();
        let r0 = z + right;
        let r1 = z - right;
        let cos_theta = r0.cbrt() + r1.cbrt();
        debug_assert!(cos_theta.abs() < 1.0, "{} {}", r0, r1);
        // cos theta = wi * wo
        // wo = cos_theta / wi;?
        let frame = TangentFrame::from_normal(wi);
        let sin_theta = (1.0 - cos_theta.powi(2)).sqrt() * if flipped { 1.0 } else { -1.0 };

        let phi = sample.y * TAU;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let vec = frame.to_world(&Vec3::new(
            sin_phi * sin_theta,
            cos_phi * sin_theta,
            cos_theta,
        ));

        let pdf = 3.0 * (1.0 + cos_theta * cos_theta) / 8.0;

        // if we treat wi as the normal

        (vec, pdf.into())
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma = self.sigma_s(lambda);
        (-sigma * (p1 - p0).norm()).exp()
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        let sigma_s = self.sigma_s(lambda);
        let dist = -(1.0 - s.x).ln() / sigma_s;
        let t = dist.min(ray.tmax); // only go as far as tmax, unless tmax is inf
                                    // did we sample the medium or did we hit the end of the ray
        let did_sample_medium = t < ray.tmax;
        let point = ray.point_at_parameter(t);
        let tr = self.tr(lambda, ray.origin, point);

        if did_sample_medium {
            (point, tr * sigma_s, true)
        } else {
            (point, tr, false)
        }
    }
}

#[cfg(test)]
mod test {
    use minifb::WindowOptions;
    use rayon::prelude::*;

    use crate::tonemap::Clamp;

    use super::*;

    enum TestMode {
        ViewPhase,
        SamplePhase,
    }
    #[test]
    fn test_phase() {
        let width = 500usize;
        let height = 500usize;
        let bounds = BOUNDED_VISIBLE_RANGE;

        let mut film = Film::new(width, height, XYZColor::BLACK);

        let a = 1.0002724293f32;
        let b = 1.64748969205f32;
        let air_curve = Curve::Cauchy { a, b };

        // we want the calculated sigma to be on the order of 5.1 * 10.0f32.powi(-31),
        // we also want to be able to calculate that medium.tr(532.0, Point3::ORIGIN, Point3::new(0.0, 0.0, 1.0))
        // should be about 10^-5

        #[rustfmt::skip]
        let medium = Rayleigh::new(
            23.0,
            air_curve
        );

        let mode = TestMode::ViewPhase;
        // let mode = TestMode::SamplePhase;
        let wi = Vec3::new(0.0, 1.0, 1.0).normalized();
        // let wi = Vec3::Z;
        let mut total_samples = 0;
        let converter = Converter::sRGB;

        let (samples_per_iteration, exposure): (usize, f32) = match mode {
            TestMode::ViewPhase => (10, 17.0),
            TestMode::SamplePhase => (1000, 10.0),
        };

        let test_wo = -wi;
        // let test_lambda = bounds.sample(random());
        let test_lambda = 532.0f32;

        let sigma_s = medium.sigma_s(test_lambda);
        println!("sigma_s = {:?}", sigma_s);

        let test_phase = medium.p(test_lambda, (0.0, 0.0, 0.0), wi, test_wo);
        println!("test phase = {:?}", test_phase);

        let test_transmittance = medium.tr(test_lambda, Point3::new(0.0, 0.0, 1.0), Point3::ORIGIN);
        println!(
            "{:?} . should be near {}",
            test_transmittance,
            1.0 - 1.0 / 100000.0
        );
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
                update_window_buffer(
                    &mut window_buffer,
                    &film,
                    &mut tonemapper,
                    converter,
                    factor,
                );
            },
        );
    }
}