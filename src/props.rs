use math::{random::random_on_unit_sphere, sample::Sample2D, vec::Vec3};
use proptest::prelude::*;

prop_compose! {
    pub fn uniform_sample()(u in 0.0..1.0f32, v in 0.0..1.0f32) -> Sample2D {
        Sample2D::new(u, v)
    }
}

prop_compose! {
    pub fn valid_ggx_roughness()(x in 0.0..1.0f32) -> f32 {
        (-(x+f32::EPSILON).ln()).recip()
    }
}

prop_compose! {
    pub fn unit_vector()(s in uniform_sample()) -> Vec3 {
        random_on_unit_sphere(s)
    }
}
