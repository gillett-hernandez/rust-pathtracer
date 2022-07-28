use crate::materials::Material;
use crate::math::*;
use crate::world::TransportMode;

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

impl Material for DiffuseLight {
    fn bsdf(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        // copy from lambertian
        if wo.z() * wi.z() > 0.0 {
            (
                SingleEnergy::new(self.bounce_color.evaluate_clamped(lambda) / PI),
                (wo.z() / PI).into(),
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
        _wi: Vec3,
    ) -> Option<Vec3> {
        // bounce, copy from lambertian
        Some(random_cosine_direction(s))
    }
    fn sample_emission(
        &self,
        point: Point3,
        normal: Vec3,
        wavelength_range: Bounds1D,
        mut scatter_sample: Sample2D,
        wavelength_sample: Sample1D,
    ) -> Option<(Ray, SingleWavelength, PDF, PDF)> {
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

    fn sample_emission_spectra(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> Option<(f32, PDF)> {
        let (sw, pdf) = self
            .emit_color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        Some((sw.lambda, pdf))
    }

    fn emission(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wi: Vec3,
    ) -> SingleEnergy {
        let cosine = wi.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            // if wi.z() > 0.0 {
            SingleEnergy::new(self.emit_color.evaluate_power(lambda) / PI)
        } else {
            SingleEnergy::ZERO
        }
    }

    // evaluate the directional pdf if the spectral power distribution
    fn emission_pdf(
        &self,
        _lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        wo: Vec3,
    ) -> PDF {
        let cosine = wo.z();
        if (cosine > 0.0 && self.sidedness == Sidedness::Forward)
            || (cosine < 0.0 && self.sidedness == Sidedness::Reverse)
            || self.sidedness == Sidedness::Dual
        {
            (cosine / PI).into()
        } else {
            0.0.into()
        }
    }
}
