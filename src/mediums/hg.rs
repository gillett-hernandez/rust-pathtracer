
use crate::math::*;

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
    pub g: Curve, // valid range: 0.0 to 2.0. actual g == g.evaluate_power(lambda) - 1.0
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
    fn sample_p(&self, lambda: f32, _uvw: (f32, f32, f32), wi: Vec3, s: Sample2D) -> (Vec3, f32) {
        // just do isomorphic as a test
        let g = self.g.evaluate_power(lambda) + 0.001 - 1.0;
        let cos_theta = if g.abs() < 0.001 {
            1.0 - 2.0 * s.x
        } else {
            let sqr = (1.0 - g * g) / (1.0 + g - 2.0 * g * s.x);
            -(1.0 + g * g - sqr * sqr) / (2.0 * g)
        };
        // println!("{} {}", cos_theta, g);

        let sin_theta = (0.0f32).max(1.0 - cos_theta * cos_theta).sqrt();
        let phi = std::f32::consts::TAU * s.y;
        let frame = TangentFrame::from_normal(wi);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let wo = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
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

        (frame.to_world(&wo), v)
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        let dist = -(1.0 - s.x).ln() / sigma_t;
        let t = dist.min(ray.tmax);
        let sampled_medium = t < ray.tmax;

        let point = ray.point_at_parameter(t);
        let tr = self.tr(lambda, ray.origin, point);
        // could add HWSS here.
        let density = if sampled_medium { sigma_t * tr } else { tr };
        let pdf = density;
        if sampled_medium {
            (point, tr * self.sigma_s.evaluate_power(lambda) / pdf, true)
        } else {
            (point, tr / pdf, false)
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        (-sigma_t * (p1 - p0).norm()).exp()
    }
}
