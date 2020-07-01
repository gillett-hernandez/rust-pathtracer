use crate::materials::*;
use crate::math::*;
// #[derive(Clone)]
// pub struct EnvironmentMap {
//     pub color: SPD,
//     pub strength: f32,
// }

#[derive(Clone)]
pub enum EnvironmentMap {
    Constant {
        color: SPD,
        strength: f32,
    },
    Sun {
        color: SPD,
        strength: f32,
        solid_angle: f32,
        uv: (f32, f32),
    },
}

impl EnvironmentMap {
    // pub const fn new(color: SPD, strength: f32) -> Self {
    //     EnvironmentMap { color, strength }
    // }
    pub fn sample_spd(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> Option<(SingleWavelength, PDF)> {
        // sample emission at a given uv when wavelength is yet to be determined.
        // used when a camera ray hits an environment map without ever having been assigned a wavelength.

        // later use uv for texture accessing
        let (mut sw, pdf) = self.color.sample_power_and_pdf(wavelength_range, sample);
        sw.energy *= self.strength;
        Some((sw, pdf))
    }

    #[allow(unused_variables)]
    pub fn emission(&self, uv: (f32, f32), lambda: f32) -> SingleEnergy {
        // evaluate emission at uv coordinate and wavelength

        let phi = PI * (2.0 * uv.0 - 1.0);
        let (y, x) = phi.sin_cos();
        let z = (PI * uv.1).cos();
        debug_assert!(lambda > 0.0);
        // SingleEnergy::new(self.color.evaluate_power(lambda))
        SingleEnergy::new(self.color.evaluate_power(lambda) * self.strength)
    }

    pub fn sample_emission(
        &self,
        world_radius: f32,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength, PDF) {
        // sample env map cdf to get light ray, based on env map strength
        let (mut sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        sw.energy *= self.strength;
        let random_direction = random_on_unit_sphere(direction_sample);
        let random_on_world = world_radius * random_on_unit_sphere(position_sample);
        let point = Point3::from(random_on_world);

        (Ray::new(point, random_direction), sw, pdf)
    }
}
