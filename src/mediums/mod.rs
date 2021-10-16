use crate::math::*;

use std::marker::{Send, Sync};

pub trait Medium {
    fn p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32;
    fn sample_p(
        &self,
        lambda: f32,
        uvw: (f32, f32, f32),
        wi: Vec3,
        sample: Sample2D,
    ) -> (Vec3, f32);
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool);
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32;
    fn emission(&self, _lambda: f32, _wo: Vec3, _uvw: (f32, f32, f32)) -> SingleEnergy {
        0.0.into()
    }
    fn sample_emission(&self, _lambda: f32, _uvw: (f32, f32, f32)) -> (Vec3, SingleEnergy) {
        (Vec3::ZERO, 0.0.into())
    }
}

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
    pub g: SPD,       // valid range: 0.0 to 2.0. actual g == g.evaluate_power(lambda) - 1.0
    pub sigma_t: SPD, // transmittance attenuation
    pub sigma_s: SPD, // scattering attenuation
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

#[derive(Clone)]
pub enum MediumEnum {
    HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous),
}

impl Medium for MediumEnum {
    fn p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, wo: Vec3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.p(lambda, uvw, wi, wo),
        }
    }
    fn sample_p(&self, lambda: f32, uvw: (f32, f32, f32), wi: Vec3, s: Sample2D) -> (Vec3, f32) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample_p(lambda, uvw, wi, s),
        }
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample(lambda, ray, s),
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.tr(lambda, p0, p1),
        }
    }
    fn emission(&self, lambda: f32, wo: Vec3, uvw: (f32, f32, f32)) -> SingleEnergy {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.emission(lambda, wo, uvw),
        }
    }
    fn sample_emission(&self, lambda: f32, uvw: (f32, f32, f32)) -> (Vec3, SingleEnergy) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample_emission(lambda, uvw),
        }
    }
}

unsafe impl Send for MediumEnum {}
unsafe impl Sync for MediumEnum {}

pub type MediumTable = Vec<MediumEnum>;
