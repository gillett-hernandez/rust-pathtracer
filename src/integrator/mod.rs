mod bdpt;
mod lt;
mod pt;
use crate::math::*;

pub trait Integrator: Sync + Send {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength;
}

pub use bdpt::BDPTIntegrator;
pub use lt::LightTracingIntegrator;
pub use pt::PathTracingIntegrator;
