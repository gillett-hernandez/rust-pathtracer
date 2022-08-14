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
    pub sigma_t: Curve, // transmittance attenuation
    pub sigma_s: Curve, // scattering attenuation
}

impl Medium for HenyeyGreensteinHomogeneous {
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
    fn sample_p(&self, lambda: f32, _uvw: (f32, f32, f32), wi: Vec3, s: Sample2D) -> (Vec3, PDF) {
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
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        let dist = -(1.0 - s.x).ln() / sigma_t;
        let t = dist.min(ray.tmax);
        let sampled_medium = t < ray.tmax;

        let point = ray.point_at_parameter(t);
        let tr = self.tr(lambda, ray.origin, point);

        if sampled_medium {
            (point, self.sigma_s.evaluate_power(lambda) / sigma_t, true)
        } else {
            (point, 1.0, false)
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        (-sigma_t * (p1 - p0).norm()).exp()
    }
}
