use math::Sidedness;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct DiffuseLight {
    // TODO: make this textured.
    pub bounce_color: Curve,
    pub emit_color: CurveWithCDF,
    pub sidedness: Sidedness,
}

impl DiffuseLight {
    pub fn new(
        bounce_color: Curve,
        emit_color: CurveWithCDF,
        sidedness: Sidedness,
    ) -> DiffuseLight {
        DiffuseLight {
            bounce_color,
            emit_color,
            sidedness,
        }
    }
    pub const NAME: &'static str = "DiffuseLight";
}

impl Material<f32, f32> for DiffuseLight {
    fn bsdf(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        if wo.z() * wi.z() > 0.0 {
            (
                self.bounce_color.evaluate_clamped(lambda) / PI,
                (wo.z().abs() / PI).into(),
            )
        } else {
            (0.0.into(), 0.0.into())
        }
    }
    fn generate(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        // bounce, copy from lambertian

        let d = random_cosine_direction(s) * wi.z().signum();
        Some(d)
    }

    fn generate_and_evaluate(
        &self,
        lambda: f32,
        uv: (f32, f32),
        transport_mode: TransportMode,
        s: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        let wi_z = wi.z();
        let d = random_cosine_direction(s) * wi_z.signum();
        let wo_z = d.z();
        (
            self.bounce_color.evaluate_clamped(lambda) / PI,
            Some(d),
            (wo_z.abs() / PI).into(),
        )
    }
    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        mut scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(
        Ray,
        SingleWavelength,
        PDF<f32, SolidAngle>,
        PDF<f32, Uniform01>,
    )> {
        // wo localized to point and normal
        let mut swap = false;
        if self.sidedness == Sidedness::Reverse {
            swap = true;
        } else if self.sidedness == Sidedness::Dual {
            if scatter_sample.x < 0.5 {
                swap = true;
                scatter_sample.x *= 2.0;
            } else {
                scatter_sample.x = (1.0 - scatter_sample.x) * 2.0;
            }
        }
        let mut local_wo = random_cosine_direction(scatter_sample) + 0.0001 * Vec3::Z;

        if swap {
            local_wo = -local_wo;
        }
        // needs to be converted to object space in a way that respects the surface normal
        let frame = TangentFrame::from_normal(normal);
        let object_wo = frame.to_world(&local_wo).normalized();
        let directional_pdf = local_wo.z().abs() / PI;
        debug_assert!(directional_pdf > 0.0, "{:?} {:?}", local_wo, object_wo);
        let (sw, pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((
            Ray::new(point, object_wo),
            sw.with_energy(sw.energy / PI),
            PDF::from(directional_pdf),
            pdf,
        ))
    }

    fn emission(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
    ) -> f32 {
        let cosine = wi.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            self.emit_color.evaluate_power(lambda) / PI
        } else {
            0.0
        }
    }

    fn emission_pdf(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wo: Vec3,
    ) -> PDF<f32, SolidAngle> {
        let cosine = wo.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            // TODO: verify output is correct for given pdf measure
            // PI.recip().into()
            (cosine / PI).into()
        } else {
            0.0.into()
        }
    }

    fn sample_emission_spectra(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF<f32, Uniform01>)> {
        let (sw, pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((sw.lambda, pdf))
    }
}
