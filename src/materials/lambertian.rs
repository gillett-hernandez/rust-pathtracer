use crate::{prelude::*, texture::EvalAt};

#[derive(Clone)]
pub struct Lambertian {
    pub texture: TexStack,
}

impl Lambertian {
    pub fn new(texture: TexStack) -> Lambertian {
        Lambertian { texture }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material<f32, f32> for Lambertian {
    fn bsdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        if wo.z() * wi.z() > 0.0 {
            (
                (self.texture.eval_at(lambda, uv).min(1.0) / PI),
                (wo.z().abs() / PI).into(),
            )
        } else {
            (0.0.into(), 0.0.into())
        }
    }
    // don't implement sample_emission, since the default implementation is what we want.
    // though perhaps it would be a good idea to panic if a the integrator tries to sample the emission of a lambertian

    // implement f

    fn generate(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        let d = random_cosine_direction(s) * wi.z().signum();
        Some(d)
    }

    fn generate_and_evaluate(
        &self,
        lambda: f32,
        uv: (f32, f32),
        _: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        let wi_z = wi.z();
        let d = random_cosine_direction(s) * wi_z.signum();
        let wo_z = d.z();
        (
            (self.texture.eval_at(lambda, uv).min(1.0) / PI),
            Some(d),
            (wo_z.abs() / PI).into(),
        )
    }
    // fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
    //     0.0
    // }
}

#[cfg(test)]
mod test {
    use math::curves::InterpolationMode;
    use math::spectral::BOUNDED_VISIBLE_RANGE;

    use crate::renderer::Film;
    use crate::texture::{Texture, Texture1};

    use super::*;
    #[test]
    fn test_material() {
        let flat = Curve::Linear {
            signal: vec![1.0],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        };
        let cdf = flat.to_cdf(BOUNDED_VISIBLE_RANGE, 5);
        let tex = Texture1 {
            curve: cdf,
            texture: Film::new(1, 1, 1.0),
            interpolation_mode: InterpolationMode::Linear,
        };
        let color = TexStack {
            textures: vec![Texture::Texture1 { 0: tex }],
        };
        let material = Lambertian::new(color);
        let wo = Vec3::Z;
        let wi = material
            .generate(
                500.0,
                (0.0, 0.0),
                TransportMode::Radiance,
                Sample2D::new_random_sample(),
                wo,
            )
            .expect("couldn't generate wi from lambert");
        let (reflectance, pdf) =
            material.bsdf(500.0, (0.0, 0.0), TransportMode::Importance, wi, wo);

        println!("{:?} -> {:?}", wi, wo);
        println!(
            "f(wi -> wo) = {} / {:?} = {}",
            reflectance,
            pdf,
            reflectance / *pdf
        );
    }
}
