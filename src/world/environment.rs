// use crate::materials::*;
use crate::math::*;

#[derive(Clone, Debug)]
pub enum EnvironmentMap {
    Constant {
        color: CDF,
        strength: f32,
    },
    Sun {
        color: CDF,
        strength: f32,
        angular_diameter: f32,
        sun_direction: Vec3,
    },
}

impl EnvironmentMap {
    // pub const fn new(color: SPD, strength: f32) -> Self {
    //     EnvironmentMap { color, strength }
    // }

    // currently unused
    // sample the spectral distribution at a env map UV
    // used when a camera ray hits an environment map without ever having been assigned a wavelength.
    // would happen when a camera ray hits an env map without bouncing on anything wavelength dependent
    // assuming that the wavelength shouldn't have already been sampled based on the camera's spectral sensitivity
    // pub fn _sample_spd(
    //     &self,
    //     _uv: (f32, f32),
    //     wavelength_range: Bounds1D,
    //     wavelength_sample: Sample1D,
    // ) -> Option<(SingleWavelength, PDF)> {
    //     // later use uv for texture accessing
    //     let (mut sw, pdf) = self
    //         .color
    //         .sample_power_and_pdf(wavelength_range, wavelength_sample);
    //     sw.energy *= self.strength;
    //     Some((sw, pdf))
    // }

    // evaluate env map given a uv and wavelength
    // used when a camera ray with a given wavelength intersects the environment map
    #[allow(unused_variables)]
    pub fn emission(&self, uv: (f32, f32), lambda: f32) -> SingleEnergy {
        // evaluate emission at uv coordinate and wavelength
        match self {
            EnvironmentMap::Constant { color, strength } => {
                debug_assert!(lambda > 0.0);
                // SingleEnergy::new(self.color.evaluate_power(lambda))
                SingleEnergy::new(color.evaluate_power(lambda) * strength)
            }
            EnvironmentMap::Sun {
                color,
                strength,
                angular_diameter,
                sun_direction,
            } => {
                let direction = uv_to_direction(uv);
                let cos = *sun_direction * direction;
                let sin = (1.0 - cos * cos).sqrt();
                if sin.abs() < (*angular_diameter / 2.0).sin() && cos > 0.0 {
                    // within solid angle
                    SingleEnergy::new(color.evaluate_power(lambda) * *strength)
                } else {
                    SingleEnergy::ZERO
                }
            }
        }
    }

    // sample a ray and wavelength based on env map CDF
    // currently disregards env CDF, since there is no env cdf.
    pub fn sample_emission(
        &self,
        world_radius: f32,
        world_center: Point3,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength, PDF, PDF) {
        // sample env map cdf to get light ray, based on env map strength
        match self {
            EnvironmentMap::Constant { color, strength } => {
                let (mut sw, pdf) = color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                sw.energy *= *strength;
                let random_direction = random_on_unit_sphere(direction_sample);
                let frame = TangentFrame::from_normal(random_direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + -random_direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (
                    Ray::new(point, random_direction),
                    sw,
                    // pdf * 1.0 / (4.0 * PI), // solid angle pdf w/ wavelength sample incorporated
                    PDF::from(1.0 / (4.0 * PI)), // solid angle pdf
                    pdf,                         // wavelength pdf
                )
            }
            EnvironmentMap::Sun {
                color,
                strength,
                angular_diameter: _,
                sun_direction: _,
            } => {
                let (mut sw, wavelength_pdf) = color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                sw.energy *= *strength;
                let (uv, directional_pdf) =
                    self.sample_env_uv_given_wavelength(direction_sample, sw.lambda);

                let direction = uv_to_direction(uv);
                let frame = TangentFrame::from_normal(direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (Ray::new(point, -direction), sw, directional_pdf, wavelength_pdf)
            }
        }
    }

    pub fn sample_direction_given_wavelength(&self, sample: Sample2D, lambda: f32) -> (Vec3, PDF) {
        // sample the env map from a specific point in the world.
        // if self.sample_env_uv_given_wavelength ever works out, use that instead here.
        let (uv, pdf) = self.sample_env_uv_given_wavelength(sample, lambda);
        let direction = uv_to_direction(uv);

        (direction, pdf)
    }

    pub fn sample_direction_and_wavelength(
        &self,
        _sample: Sample2D,
        _wavelength_range: Bounds1D,
        _wavelength_sample: Sample1D,
    ) -> (Vec3, PDF) {
        unimplemented!()
    }

    // sample env UV given a wavelength, based on env CDF for a specific wavelength. might be hard to evaluate, or nearly impossible.
    // would be used when sampling the environment from an eye path, such as in PT or BDPT, given a wavelength
    pub fn sample_env_uv_given_wavelength(
        &self,
        sample: Sample2D,
        _lambda: f32,
    ) -> ((f32, f32), PDF) {
        self.sample_env_uv(sample)
        // match self {
        //     EnvironmentMap::Constant { color, strength } => self.sample_env_uv(sample),
        //     EnvironmentMap::Sun {
        //         color,
        //         strength,
        //         angular_diameter,
        //         sun_direction,
        //     } => {}
        // }

        // however because that's unimplemented for now, lets just return `sample_env_uv`
    }

    // sample env UV, based on env luminosity CDF (w/o prescribed wavelength)
    pub fn sample_env_uv(&self, sample: Sample2D) -> ((f32, f32), PDF) {
        // samples env CDF to find bright luminosity spikes. returns UV of those spots.
        // CDF for this situation can be stored as the Y values of the XYZ representation, as a greyscale image potentially.
        // consider summed area table as well.
        match self {
            EnvironmentMap::Constant {
                color: _color,
                strength: _strength,
            } => ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI))),
            EnvironmentMap::Sun {
                color: _color,
                strength: _strength,
                 angular_diameter,
                sun_direction,
            } => {
                let local_wo = Vec3::Z + (*angular_diameter / 2.0).sin() * random_in_unit_disk(sample);
                let sun_direction = *sun_direction;
                let frame = TangentFrame::from_normal(sun_direction);
                let direction = frame.to_world(&local_wo);
                (
                    direction_to_uv(direction.normalized()),
                    PDF::from(1.0 / (2.0 * PI * (1.0 - *angular_diameter))),
                    // 1.0.into()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves;
    #[test]
    fn test_sample_emission() {
        let env_map = EnvironmentMap::Constant {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
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

    #[test]
    fn test_sample_emission_sun() {
        let env_map = EnvironmentMap::Sun {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
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

    #[test]
    fn test_uv_to_direction_and_back() {
        let direction = Vec3::new(1.0, 1.0, 1.0).normalized();
        println!("{:?}", direction);
        let uv = direction_to_uv(direction);
        println!("{:?}", uv);
        let direction_again = uv_to_direction(uv);
        println!("{:?}", direction_again);
    }

    #[test]
    fn test_sample_env_map (){
        let env_map = EnvironmentMap::Sun {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };

        env_map.sample_direction_given_wavelength(Sample2D::new_random_sample(), 500.0);
    }
}
