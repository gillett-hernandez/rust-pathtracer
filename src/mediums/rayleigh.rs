use crate::math::*;

use super::Medium;

#[derive(Clone)]
pub struct Rayleigh {
    pub particle_size_factor: f32,
    pub ior: Curve,
}

// FIXME: correct scale of certain constants such that a reasonable sigma and phase are outputted, i.e.
// such that they don't require a massive scale factor to cause things to actually render correctly.
// since sigma is the "scattering cross section", maybe figure out exactly how to convert that to the correct sigma for use in the integrator


impl Rayleigh {
    fn ior_factor(&self, lambda: f32) -> f32 {
        let n = self.ior.evaluate(lambda);
        let n_2 = n * n;
        let numerator = n_2 - 1.0;
        let denominator = n_2 + 2.0;
        (numerator / denominator).powi(2)
    }
    fn sigma_s(&self, lambda: f32) -> f32 {
        let ior_factor = self.ior_factor(lambda);
        let factor_particle = self.particle_size_factor.powi(6);
        let lambda_factor = lambda.recip().powi(4);
        2.0 / 3.0 * PI.powi(5) * ior_factor * factor_particle * lambda_factor
    }
}

impl Medium for Rayleigh {
    fn p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32 {
        // I = I_0 * (1+cos^2(theta)) / 2R^2 * (2pi/lambda)^4 * ((n^2-1) / (n^2 + 2))^2 * (d/2)^6
        // where:
        //      R = distance to observer
        //      theta = scattering angle
        //      n = index of refraction
        //      d = averaged particle diameter

        let factor_ior = self.ior_factor(lambda);

        let factor_particle = (self.particle_size_factor / 2.0).powi(6);
        let factor_lambda = (2.0 * PI / lambda).powi(4);
        let cos_squared = wo.z().abs().powi(2);
        (1.0 + cos_squared) * factor_lambda * factor_ior * factor_particle
    }
    fn sample_p(
        &self,
        lambda: f32,
        uvw: (f32, f32, f32),
        wi: Vec3,
        sample: Sample2D,
    ) -> (Vec3, f32) {
        (Vec3::Z, 0.0)
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        let sigma_s = self.sigma_s(lambda);
        (Point3::ZERO, 0.0, false)
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma_s = self.sigma_s(lambda);
        0.0
    }
}
