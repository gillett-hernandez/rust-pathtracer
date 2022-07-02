use crate::materials::Material;
use crate::math::*;
use crate::texture::TexStack;
use crate::world::TransportMode;

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

impl Material for Lambertian {
    fn generate(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        s: Sample2D,
        _wi: Vec3,
    ) -> Option<Vec3> {
        Some(random_cosine_direction(s))
    }
    // don't implement sample_emission, since the default implementation is what we want.
    // though perhaps it would be a good idea to panic if a the integrator tries to sample the emission of a lambertian

    // implement f

    fn bsdf(
        &self,
        lambda: f32,
        uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (
                SingleEnergy::new(self.texture.eval_at(lambda, uv).min(1.0) / PI),
                (cosine / PI).into(),
            )
        } else {
            (0.0.into(), 0.0.into())
        }
    }
    // fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
    //     SingleEnergy::ZERO
    // }
}

#[cfg(test)]
mod test {
    use math::curves::InterpolationMode;
    use math::spectral::BOUNDED_VISIBLE_RANGE;
    use math::*;

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
            "f(wi -> wo) = {} / {} = {}",
            reflectance.0,
            pdf.0,
            reflectance.0 / pdf.0
        );
    }
}
