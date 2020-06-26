mod bdpt;
mod lt;
mod pt;
use crate::math::*;

pub trait SamplerIntegrator: Sync + Send {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength;
}

pub trait Integrator: Send + Sync {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        samples: &mut Vec<(SingleWavelength, (usize, usize))>,
    );
}

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;
