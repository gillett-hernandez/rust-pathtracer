use crate::prelude::*;
use crate::texture::EvalAt;
use crate::world::importance_map::ImportanceMap;
use std::ops::Mul;

#[derive(Clone)]
pub enum EnvironmentMap {
    Constant {
        color: CurveWithCDF,
        strength: f32,
    },
    Sun {
        color: CurveWithCDF,
        strength: f32,
        angular_diameter: f32,
        sun_direction: Vec3,
    },
    // proposal: generate importance maps for each CurveWithCDF associated with the HDR texture.
    // select importance map and Curve randomly based on max luminance,
    // i.e. if there's a texture with a low max luminance compared to other env textures, it should not be selected for importance sampling very often.
    HDR {
        texture: TexStack,
        importance_map: ImportanceMap,
        rotation: Transform3,
        strength: f32,
    },
}

impl EnvironmentMap {
    // pub const fn new(color: Curve, strength: f32) -> Self {
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

    pub fn emission<T>(&self, uv: (f32, f32), lambda: T) -> T
    where
        CurveWithCDF: SpectralPowerDistributionFunction<T>,
        TexStack: EvalAt<T>,
        T: Field + Mul<f32, Output = T>,
    {
        // how to express trait bounds for this?
        // CurveWithCDF needs to implement SPDF, which it does for f32 and f32x4.
        // evaluate emission at uv coordinate and wavelength
        match self {
            EnvironmentMap::Constant { color, strength } => {
                color.evaluate_power(lambda) * *strength
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
                    color.evaluate_power(lambda) * *strength
                } else {
                    T::ZERO
                }
            }
            EnvironmentMap::HDR {
                texture,
                rotation,
                strength,
                ..
            } => {
                let direction = uv_to_direction(uv);
                // use to_local to transform ray direction to "uv space"
                let new_direction = rotation.to_local(direction);
                let uv = direction_to_uv(new_direction);
                let result = texture.eval_at(lambda, uv) * *strength;
                result
            }
        }
    }

    pub fn sample_emission(
        &self,
        world_radius: f32,
        world_center: Point3,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (
        Ray,
        SingleWavelength,
        PDF<f32, SolidAngle>,
        PDF<f32, Uniform01>,
    ) {
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
                let (mut sw, wavelength_pdf) =
                    color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                sw.energy *= *strength;
                let (uv, directional_pdf) =
                    self.sample_env_uv_given_wavelength(direction_sample, sw.lambda);

                let direction = uv_to_direction(uv);
                let frame = TangentFrame::from_normal(direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (
                    Ray::new(point, -direction),
                    sw,
                    directional_pdf,
                    wavelength_pdf,
                )
            }
            EnvironmentMap::HDR {
                // rotation is already taken into account when calling sample_env_uv, so ignore it
                // importance map as well
                texture,
                strength,
                rotation: _,
                ..
            } => {
                // let (mut sw, wavelength_pdf) =
                //     color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                // sw.energy *= *strength;
                let (uv, directional_pdf) = self.sample_env_uv(direction_sample);
                let (mut sw, wavelength_pdf) = (
                    SingleWavelength::new_from_range(wavelength_sample.x, wavelength_range),
                    // 1.0 / wavelength_range.span(),
                    1.0,
                );
                sw.energy = texture.eval_at(sw.lambda, uv) * strength;

                // NOTE: sample_env_uv already translates uv through `rotation`, so don't do it again here.
                let direction = uv_to_direction(uv);
                let frame = TangentFrame::from_normal(direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                // reverse direction because `direction` points from the origin to the point on the environment, so `-direction` points from the environment onto the scene
                (
                    Ray::new(point, -direction),
                    sw,
                    directional_pdf,
                    wavelength_pdf.into(),
                )
            }
        }
    }

    pub fn pdf_for(&self, uv: (f32, f32)) -> PDF<f32, SolidAngle> {
        // pdf is solid angle pdf, since projected solid angle doesn't apply to environments.
        match self {
            EnvironmentMap::Constant { .. } => (4.0 * PI).recip().into(),
            EnvironmentMap::Sun {
                angular_diameter,
                sun_direction,
                ..
            } => {
                // TODO: verify this pdf integrates to one over the sphere
                let direction = uv_to_direction(uv);
                let cos = *sun_direction * direction;
                let sin = (1.0 - cos * cos).sqrt();
                if sin.abs() < (*angular_diameter / 2.0).sin() && cos > 0.0 {
                    // within solid angle
                    PDF::from(1.0 / (2.0 * PI * (1.0 - angular_diameter.cos())))
                } else {
                    0.0.into()
                }
            }
            EnvironmentMap::HDR {
                rotation,
                importance_map,
                ..
            } => {
                if let ImportanceMap::Baked {
                    data, marginal_cdf, ..
                } = importance_map
                {
                    let direction = uv_to_direction(uv);
                    let new_direction = rotation.to_local(direction);
                    let uv = direction_to_uv(new_direction);

                    // pdf returned from this branch of this function currently is not a pdf wrt solid angle,
                    // but rather is a pdf wrt uv space [0,1) x [0,1). need to use jacobian to convert to solid angle.

                    // theta(u, v) = (u - 0.5) * 2 * pi
                    // phi(u, v) = pi * v

                    // jacobian determinant is
                    // [ 2pi , 0
                    //  0, pi ] = 2pi^2

                    // jacobian determinant from theta, phi to solid angle is just sin(phi)

                    // thus the combined jacobian from uv to solid angle is 2pi^2 * sin(pi * v)

                    PDF::from(
                        marginal_cdf.evaluate_power(uv.0)
                            * data[(uv.0.clamp(0.0, 1.0 - std::f32::EPSILON) * data.len() as f32)
                                as usize]
                                .evaluate_power(uv.1)
                            * (2.0 * PI * PI * (PI * uv.1).sin() + 0.001)
                            + 0.001,
                    )
                } else {
                    PDF::from((4.0 * PI).recip())
                }
            }
        }
    }

    pub fn sample_direction_given_wavelength(
        &self,
        sample: Sample2D,
        lambda: f32,
    ) -> (Vec3, PDF<f32, SolidAngle>) {
        let (uv, pdf) = self.sample_env_uv_given_wavelength(sample, lambda);
        let direction = uv_to_direction(uv);

        (direction, pdf)
    }

    pub fn sample_direction_and_wavelength(
        &self,
        _sample: Sample2D,
        _wavelength_range: Bounds1D,
        _wavelength_sample: Sample1D,
    ) -> (Vec3, PDF<f32, SolidAngle>) {
        // TODO
        todo!()
    }

    // sample env UV given a wavelength, based on env CDF for a specific wavelength. might be hard to evaluate, or nearly impossible.
    // would be used when sampling the environment from an eye path, such as in PT or BDPT, given a wavelength
    pub fn sample_env_uv_given_wavelength<T>(
        &self,
        sample: Sample2D,
        lambda: T,
    ) -> ((f32, f32), PDF<f32, SolidAngle>)
    where
        CurveWithCDF: SpectralPowerDistributionFunction<T>,
        TexStack: EvalAt<T>,
        T: Field + Mul<f32, Output = T>,
    {
        match self {
            EnvironmentMap::Constant { .. } => self.sample_env_uv(sample),
            EnvironmentMap::Sun { .. } => self.sample_env_uv(sample),
            EnvironmentMap::HDR { .. } => self.sample_env_uv(sample),
        }

        // however because that's unimplemented for now, lets just return `sample_env_uv`
    }

    // sample env UV, based on env luminosity CDF (w/o prescribed wavelength)
    pub fn sample_env_uv(&self, sample: Sample2D) -> ((f32, f32), PDF<f32, SolidAngle>) {
        // samples env CDF to find bright luminosity spikes. returns UV of those spots.
        // CDF for this situation can be stored as the Y values of the XYZ representation, as a greyscale image potentially.
        // consider summed area table as well.
        match self {
            EnvironmentMap::Constant { .. } => ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI))),
            EnvironmentMap::Sun {
                angular_diameter,
                sun_direction,
                ..
            } => {
                let local_wo =
                    Vec3::Z + (*angular_diameter / 2.0).sin() * random_in_unit_disk(sample);
                let sun_direction = *sun_direction;
                let frame = TangentFrame::from_normal(sun_direction);
                let direction = frame.to_world(&local_wo);
                (
                    direction_to_uv(direction.normalized()),
                    PDF::from(1.0 / (2.0 * PI * (1.0 - angular_diameter.cos()))),
                    // 1.0.into()
                )
            }
            EnvironmentMap::HDR {
                rotation,
                importance_map,
                ..
            } => {
                if let ImportanceMap::Baked { .. } = importance_map {
                    // inverse transform sample the vertical cdf
                    let (uv, pdf) = importance_map.sample_uv(sample);

                    let (row_pdf, column_pdf) = pdf;
                    let local_wo = uv_to_direction(uv);
                    let new_wo = rotation.to_world(local_wo);
                    let uv = direction_to_uv(new_wo);
                    (
                        uv,
                        PDF::from(
                            *row_pdf * *column_pdf * (2.0 * PI * PI * (PI * uv.1).sin() + 0.001)
                                + 0.001,
                        ),
                    )
                    // ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI)))
                } else {
                    ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI)))
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use math::spectral::BOUNDED_VISIBLE_RANGE;

    use super::*;
    use crate::curves;
    #[test]
    fn test_sample_emission() {
        let env_map = EnvironmentMap::Constant {
            color: curves::blackbody(5500.0, 40.0).to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            strength: 1.0,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
            Sample2D::new_random_sample(),
            Sample2D::new_random_sample(),
            BOUNDED_VISIBLE_RANGE,
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
            color: curves::blackbody(5500.0, 40.0).to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
            Sample2D::new_random_sample(),
            Sample2D::new_random_sample(),
            BOUNDED_VISIBLE_RANGE,
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
    fn test_sample_env_map() {
        let env_map = EnvironmentMap::Sun {
            color: curves::blackbody(5500.0, 40.0).to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };

        env_map.sample_direction_given_wavelength(Sample2D::new_random_sample(), 500.0);
    }
}
