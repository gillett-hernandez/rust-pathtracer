use crate::materials::*;
use crate::math::*;
#[derive(Clone)]
pub struct EnvironmentMap {
    pub color: SPD,
    pub strength: f32,
}

impl EnvironmentMap {
    pub const fn new(color: SPD, strength: f32) -> Self {
        EnvironmentMap { color, strength }
    }
    pub fn uv_to_direction(uv: (f32, f32)) -> Vec3 {
        let phi = PI * (2.0 * uv.0 - 1.0);
        let (y, x) = phi.sin_cos();
        let z = (PI * uv.1).cos();
        Vec3::new(x, y, z)
    }

    pub fn direction_to_uv(direction: Vec3) -> (f32, f32) {
        let u = (PI + direction.y().atan2(direction.x())) / (2.0 * PI);
        let v = direction.z().acos() / PI;
        (u, v)
    }

    // currently unused
    // sample the spectral distribution at a env map UV
    // used when a camera ray hits an environment map without ever having been assigned a wavelength.
    // would happen when a camera ray hits an env map without bouncing on anything wavelength dependent
    // assuming that the wavelength shouldn't have already been sampled based on the camera's spectral sensitivity
    pub fn _sample_spd(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(SingleWavelength, PDF)> {
        // later use uv for texture accessing
        let (mut sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        sw.energy *= self.strength;
        Some((sw, pdf))
    }

    // evaluate env map given a uv and wavelength
    // used when a camera ray with a given wavelength intersects the environment map
    #[allow(unused_variables)]
    pub fn emission(&self, uv: (f32, f32), lambda: f32) -> SingleEnergy {
        // evaluate emission at uv coordinate and wavelength

        debug_assert!(lambda > 0.0);
        // SingleEnergy::new(self.color.evaluate_power(lambda))
        SingleEnergy::new(self.color.evaluate_power(lambda) * self.strength)
    }

    // sample a ray and wavelength based on env map CDF
    // currently disregards env CDF, since there is no env cdf.
    pub fn sample_emission(
        &self,
        world_radius: f32,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength, PDF) {
        // sample env map cdf to get light ray, based on env map strength
        let (mut sw, _pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        sw.energy *= self.strength;
        let random_direction = random_on_unit_sphere(direction_sample);
        let frame = TangentFrame::from_normal(random_direction);
        let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
        let point =
            Point3::from(-random_direction * world_radius) + frame.to_world(&random_on_normal_disk);

        (
            Ray::new(point, random_direction),
            sw,
            // pdf * 1.0 / (4.0 * PI), // solid angle pdf w/ wavelength sample incorporated
            PDF::from(1.0 / (4.0 * PI)), // solid angle pdf
        )
    }

    pub fn sample_direction_given_wavelength(&self, sample: Sample2D, _lambda: f32) -> Vec3 {
        // sample the env map from a specific point in the world.
        // if self.sample_env_uv_given_wavelength ever works out, use that instead here.
        let uv = self.sample_env_uv(sample);
        let direction = EnvironmentMap::uv_to_direction(uv);

        direction
    }

    pub fn sample_direction_and_wavelength(
        &self,
        _sample: Sample2D,
        _wavelength_range: Bounds1D,
        _wavelength_sample: Sample1D,
    ) -> Vec3 {
        unimplemented!()
    }

    // sample env UV given a wavelength, based on env CDF for a specific wavelength. might be hard to evaluate, or nearly impossible.
    // would be used when sampling the environment from an eye path, such as in PT or BDPT, given a wavelength
    pub fn sample_env_uv_given_wavelength(&self, sample: Sample2D, _lambda: f32) -> (f32, f32) {
        // should sample env CDF to find bright power spikes for a given wavelength and return the UV of those spots.

        // however because that's unimplemented for now, lets just return `sample_env_uv`
        self.sample_env_uv(sample)
    }

    // sample env UV, based on env luminosity CDF (w/o prescribed wavelength)
    pub fn sample_env_uv(&self, sample: Sample2D) -> (f32, f32) {
        // samples env CDF to find bright luminosity spikes. returns UV of those spots.
        // CDF for this situation can be stored as the Y values of the XYZ representation, as a greyscale image potentially.
        // consider summed area table as well.
        (sample.x, sample.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves;
    #[test]
    fn test_sample_emission() {
        let env_map = EnvironmentMap::new(curves::blackbody(5500.0, 40.0), 1.0);
        let (ray, sw, pdf) = env_map.sample_emission(
            4.0,
            Sample2D::new_random_sample(),
            Sample2D::new_random_sample(),
            curves::EXTENDED_VISIBLE_RANGE,
            Sample1D::new_random_sample(),
        );
        println!("{:?} {:?} {:?}", ray, sw, pdf);
        let Ray {
            origin,
            direction,
            time: _,
            tmax: _,
        } = ray;

        let dir_toward_world_origin = Point3::ORIGIN - origin;
        let dot = dir_toward_world_origin * direction;
        println!("{}", dot);
    }
}
