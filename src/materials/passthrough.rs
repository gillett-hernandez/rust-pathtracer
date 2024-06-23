use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct PassthroughFilter {
    pub color: Curve,
    pub outer_medium_id: MediumId,
    pub inner_medium_id: MediumId,
}

impl PassthroughFilter {
    pub fn new(
        color: Curve,
        outer_medium_id: MediumId,
        inner_medium_id: MediumId,
    ) -> PassthroughFilter {
        PassthroughFilter {
            color,
            outer_medium_id,
            inner_medium_id,
        }
    }
    pub const NAME: &'static str = "PassthroughFilter";
}

// essentially a pseudo-transparent film material. does not alter the direction when scattering, and
impl Material<f32, f32> for PassthroughFilter {
    fn bsdf(
        &self,
        lambda: f32,
        _uv: UV,
        _transport_mode: TransportMode,
        _wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        // TODO: maybe have this switch between wo and wi based on transport_mode?
        // TODO: assess whether this pdf needs to be 0 if wi != -wo
        (self.color.evaluate(lambda) / wo.z().abs(), 1.0.into())
    }
    fn generate(
        &self,
        _lambda: f32,
        _uv: UV,
        _transport_mode: TransportMode,
        _s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        Some(-wi)
    }
    fn outer_medium_id(&self, _uv: UV) -> MediumId {
        self.outer_medium_id
    }
    fn inner_medium_id(&self, _uv: UV) -> MediumId {
        self.inner_medium_id
    }

    fn generate_and_evaluate(
        &self,
        lambda: f32,
        uv: UV,
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        (
            self.color.evaluate(lambda) / wi.z().abs(),
            Some(-wi),
            1.0.into(),
        )
    }
}

unsafe impl Send for PassthroughFilter {}
unsafe impl Sync for PassthroughFilter {}
