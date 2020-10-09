use crate::materials::Material;
use crate::math::*;
use crate::texture::TexStack;
use crate::world::TransportMode;

#[derive(Clone)]
pub struct Isotropic {
    pub color: SPD,
}

impl Isotropic {
    pub fn new(color: SPD) -> Isotropic {
        Isotropic { color }
    }
    pub const NAME: &'static str = "Isotropic";
}

impl Material for Isotropic {
    fn is_medium(&self) -> bool {
        true
    }
    fn eval_phase(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (SingleEnergy, PDF) {
        (
            (self.color.evaluate(lambda) * (4.0 * PI).recip()).into(),
            (4.0 * PI).recip().into(),
        )
    }
    fn sample_phase(&self, lambda: f32, wi: Vec3, s: Sample2D) -> (Vec3, PDF) {
        (random_on_unit_sphere(s), (4.0 * PI).recip().into())
    }
    fn sample_tr(
        &self,
        lambda: f32,
        time_bounds: Bounds1D,
        s: Sample1D,
    ) -> (f32, SingleEnergy, PDF) {
        let tr_at_end = self.tr(lambda, time_bounds.upper - time_bounds.lower);
        if s.x < tr_at_end.0 {
            (time_bounds.upper, tr_at_end, tr_at_end.0.into())
        } else {
            let t = time_bounds.lerp((s.x - time_bounds.lower) / time_bounds.span());
            (
                t,                                                 // time
                self.tr(lambda, t - time_bounds.lower),            // Tr
                ((1.0 - tr_at_end.0) / time_bounds.span()).into(), //PDF of scattering within time bounds
            )
        }
    }
    fn tr(&self, lambda: f32, distance: f32) -> SingleEnergy {
        (-distance).exp().into()
    }
}
