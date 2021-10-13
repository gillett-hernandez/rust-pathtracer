use crate::materials::Material;
use crate::math::*;
use crate::world::TransportMode;

#[derive(Clone)]
pub struct PassthroughFilter {
    pub color: SPD,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
}

impl PassthroughFilter {
    pub fn new(color: SPD, outer_medium_id: usize, inner_medium_id: usize) -> PassthroughFilter {
        PassthroughFilter {
            color,
            outer_medium_id,
            inner_medium_id,
        }
    }
    pub const NAME: &'static str = "PassthroughFilter";
}

impl Material for PassthroughFilter {
    fn generate(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        _s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        Some(-wi)
    }
    fn bsdf(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        _wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        // TODO: maybe have this switch between wo and wi based on transport_mode
        (
            SingleEnergy::from(self.color.evaluate(lambda) / wo.z().abs()),
            1.0.into(),
        )
    }
    fn outer_medium_id(&self, _uv: (f32, f32)) -> usize {
        self.outer_medium_id
    }
    fn inner_medium_id(&self, _uv: (f32, f32)) -> usize {
        self.inner_medium_id
    }
}

unsafe impl Send for PassthroughFilter {}
unsafe impl Sync for PassthroughFilter {}
