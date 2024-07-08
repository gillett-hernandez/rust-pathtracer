use crate::prelude::*;

pub fn reflect(wi: Vec3, normal: Vec3) -> Vec3 {
    let wi = -wi;
    (wi - 2.0 * (wi * normal) * normal).normalized()
}

pub fn refract(wi: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = wi * normal;
    let sin_2_theta_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin_2_theta_t = eta * eta * sin_2_theta_i;
    if sin_2_theta_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin_2_theta_t).sqrt();
    Some((-wi * eta + normal * (eta * cos_i - cos_t)).normalized())
}

pub fn fresnel_dielectric(eta_i: f32, eta_t: f32, cos_i: f32) -> f32 {
    // let swapped = if cos_i < 0 {
    //     cos_i = -cos_i;
    //     true
    // } else {
    //     false
    // };
    // let (eta_i, eta_t) = if swapped {
    //     (eta_t, eta_i)
    // } else {
    //     (eta_i, eta_t)
    // };
    let cos_i = cos_i.clamp(-1.0, 1.0);

    let (cos_i, eta_i, eta_t) = if cos_i < 0.0 {
        (-cos_i, eta_t, eta_i)
    } else {
        (cos_i, eta_i, eta_t)
    };

    let sin_t = eta_i / eta_t * (0.0f32).max(1.0 - cos_i * cos_i).sqrt();
    let cos_t = (0.0f32).max(1.0 - sin_t * sin_t).sqrt();
    let ei_ct = eta_i * cos_t;
    let et_ci = eta_t * cos_i;
    let ei_ci = eta_i * cos_i;
    let et_ct = eta_t * cos_t;
    let r_par = (et_ci - ei_ct) / (et_ci + ei_ct);
    let r_perp = (ei_ci - et_ct) / (ei_ci + et_ct);
    (r_par * r_par + r_perp * r_perp) / 2.0
}

pub fn fresnel_conductor(eta_i: f32, eta_t: f32, k_t: f32, cos_theta_i: f32) -> f32 {
    let cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    // handle dielectrics

    let (cos_theta_i, eta_i, eta_t) = if cos_theta_i < 0.0 {
        (-cos_theta_i, eta_t, eta_i)
    } else {
        (cos_theta_i, eta_i, eta_t)
    };

    // onto the full equations

    let eta = eta_t / eta_i;
    let etak = k_t / eta_i;

    let cos_theta_i2 = cos_theta_i * cos_theta_i;
    let sin_theta_i2 = 1.0 - cos_theta_i2;
    let eta2 = eta * eta;
    let etak2 = etak * etak;

    let t0 = eta2 - etak2 - sin_theta_i2;
    debug_assert!(t0 * t0 + eta2 * etak2 > 0.0);
    let a2plusb2 = (t0 * t0 + eta2 * etak2 * 4.0).sqrt();
    let t1 = a2plusb2 + cos_theta_i2;
    debug_assert!(a2plusb2 + t0 > 0.0);
    let a = ((a2plusb2 + t0) * 0.5).sqrt();
    let t2 = a * cos_theta_i * 2.0;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = a2plusb2 * cos_theta_i2 + sin_theta_i2 * sin_theta_i2;
    let t4 = t2 * sin_theta_i2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    (rs + rp) / 2.0
}

fn ggx_d(alpha: f32, wm: Vec3) -> f32 {
    let slope = (wm.x() / alpha, wm.y() / alpha);
    let slope2 = (slope.0 * slope.0, slope.1 * slope.1);
    let t = wm.z() * wm.z() + slope2.0 + slope2.1;
    debug_assert!(t > 0.0, "{:?} {:?}", wm, slope2);
    let a2 = alpha * alpha;
    let t2 = t * t;
    let aatt = a2 * t2;
    debug_assert!(aatt > 0.0, "{} {} {:?}", alpha, t, wm);
    1.0 / (PI * aatt)
}

fn ggx_lambda(alpha: f32, w: Vec3) -> f32 {
    if w.z() == 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let w2 = Vec3(w.0 * w.0);
    let c = 1.0 + (a2 * w2.x() + a2 * w2.y()) / w2.z(); // replace a2 with Vec2 for anistropy
    c.sqrt() * 0.5 - 0.5
}

fn ggx_g(alpha: f32, wi: Vec3, wo: Vec3) -> f32 {
    let bottom = 1.0 + ggx_lambda(alpha, wi) + ggx_lambda(alpha, wo);
    debug_assert!(bottom != 0.0);
    bottom.recip()
}

fn ggx_vnpdf(alpha: f32, wi: Vec3, wh: Vec3) -> f32 {
    let inv_gl = 1.0 + ggx_lambda(alpha, wi);
    debug_assert!(wh.0.is_finite().all());
    (ggx_d(alpha, wh) * (wi * wh).abs()) / (inv_gl * wi.z().abs())
}

fn ggx_vnpdf_no_d(alpha: f32, wi: Vec3, wh: Vec3) -> f32 {
    ((wi * wh) / ((1.0 + ggx_lambda(alpha, wi)) * wi.z())).abs()
}

// fn ggx_pdf(alpha: f32, _wi: Vec3, wh: Vec3) -> f32 {
//     ggx_d(alpha, wh) * wh.z().abs()
// }

fn sample_vndf(alpha: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
    let Sample2D { x, y } = sample;
    let v = Vec3::new(alpha * wi.x(), alpha * wi.y(), wi.z()).normalized();

    let t1 = if v.z() < 0.9999 {
        v.cross(Vec3::Z).normalized()
    } else {
        Vec3::X
    };
    let t2 = t1.cross(v);
    debug_assert!(v.0.is_finite().all(), "{:?}", v);
    debug_assert!(t1.0.is_finite().all(), "{:?}", t1);
    debug_assert!(t2.0.is_finite().all(), "{:?}", t2);
    let a = 1.0 / (1.0 + v.z());
    let r = x.sqrt();
    debug_assert!(r.is_finite(), "{}", x);
    let phi = if y < a {
        y / a * PI
    } else {
        PI + (y - a) / (1.0 - a) * PI
    };

    let (sin_phi, cos_phi) = phi.sin_cos();
    debug_assert!(sin_phi.is_finite() && cos_phi.is_finite(), "{:?}", phi);
    let p1 = r * cos_phi;
    let p2 = r * sin_phi * if y < a { 1.0 } else { v.z() };
    let value = 1.0 - p1 * p1 - p2 * p2;
    let n = p1 * t1 + p2 * t2 + value.max(0.0).sqrt() * v;

    debug_assert!(
        n.0.is_finite().all(),
        "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
        n,
        p1,
        t1,
        p2,
        t2,
        v
    );
    Vec3::new(alpha * n.x(), alpha * n.y(), n.z().max(0.0)).normalized()
}

fn sample_wh(alpha: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
    // normal invert mark
    let flip = wi.z() < 0.0;
    let wh = sample_vndf(alpha, if flip { -wi } else { wi }, sample);
    if flip {
        -wh
    } else {
        wh
    }
}

#[derive(Debug, Clone)]
pub struct GGX {
    pub alpha: f32, // todo: let this be controlled by a texture
    pub eta: Curve,
    pub eta_o: Curve,
    pub kappa: Curve,
    pub outer_medium_id: MediumId,
    pub inner_medium_id: MediumId,
    metallic: bool,
}

impl GGX {
    pub const NAME: &'static str = "GGX";
    pub fn new(
        roughness: f32,
        eta: Curve,
        eta_o: Curve,
        kappa: Curve,
        outer_medium_id: MediumId,
        inner_medium_id: MediumId,
    ) -> Self {
        debug_assert!(roughness > 0.0);

        let metallic = kappa.evaluate_integral(BOUNDED_VISIBLE_RANGE, 100, false) > 0.0;
        warn!(
            "constructing new ggx with metallic bool set to {}",
            metallic
        );
        GGX {
            alpha: roughness,
            eta,
            eta_o,
            kappa,
            metallic,
            outer_medium_id,
            inner_medium_id,
        }
    }

    fn reflectance(&self, eta_outer: f32, eta_inner: f32, kappa: f32, cos_theta_i: f32) -> f32 {
        if !self.metallic {
            fresnel_dielectric(eta_outer, eta_inner, cos_theta_i)
        } else {
            fresnel_conductor(eta_outer, eta_inner, kappa, cos_theta_i)
        }
    }

    fn reflectance_probability(
        &self,
        eta_outer: f32,
        eta_inner: f32,
        kappa: f32,
        cos_theta_i: f32,
    ) -> f32 {
        if !self.metallic {
            self.reflectance(eta_outer, eta_inner, kappa, cos_theta_i)
                .clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
    fn eta_rel(&self, eta_outer: f32, eta_inner: f32, wi: Vec3) -> f32 {
        // TODO: determine if this should take cos_i rather than wi,
        // so that we can pass in ndotv instead of wi, allowing the choice of eta_rel to take into account
        // what wh * wi looks like
        if wi.z() < 0.0 {
            eta_outer / eta_inner
        } else {
            eta_inner / eta_outer
        }
    }
}

impl Material<f32, f32> for GGX {
    fn bsdf(
        &self,
        lambda: f32,
        _uv: UV,
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        let wi = wi.normalized();
        let same_hemisphere = wi.z() * wo.z() > 0.0;

        let g = (wi.z() * wo.z()).abs();

        if g == 0.0 {
            return (f32::ZERO, 0.0.into());
        }

        let cos_i = wi.z();

        let mut glossy = f32::ZERO;
        let mut transmission = f32::ZERO;
        let mut glossy_pdf = 0.0;
        let mut transmission_pdf = 0.0;
        let eta_inner = self.eta.evaluate_power(lambda);
        let eta_outer = self.eta_o.evaluate_power(lambda);
        let kappa = if self.metallic {
            self.kappa.evaluate_power(lambda)
        } else {
            0.0
        };
        if same_hemisphere {
            let mut wh = (wo + wi).normalized();
            // normal invert mark
            if wh.z() < 0.0 {
                wh = -wh;
            }

            let ndotv = wi * wh;
            let refl = self.reflectance(eta_outer, eta_inner, kappa, ndotv);
            debug_assert!(wh.0.is_finite().all());
            let ggxd = ggx_d(self.alpha, wh);
            let ggxg = ggx_g(self.alpha, wi, wo);
            glossy = refl * (0.25 / g) * ggxd * ggxg;
            if ndotv.abs() == 0.0 {
                glossy_pdf = 0.0;
            } else {
                glossy_pdf = ggx_vnpdf(self.alpha, wi, wh) * 0.25 / ndotv.abs();
            }
            debug_assert!(glossy_pdf.is_finite(), "{:?} {}", self.alpha, ndotv);
            if glossy_pdf == 0.0 {
                trace!(
                    "same hemisphere, {:?} ndv: {} refl: {} d:{} g:{}, g:{} p:{}",
                    wh,
                    ndotv,
                    refl,
                    ggxd,
                    ggxg,
                    glossy,
                    glossy_pdf,
                );
            }
        } else if !self.metallic {
            let eta_rel = self.eta_rel(eta_outer, eta_inner, wi);

            let ggxg = ggx_g(self.alpha, wi, wo);
            debug_assert!(
                wi.0.is_finite().all() && wo.0.is_finite().all(),
                "{:?} {:?} {:?} {:?}",
                wi,
                wo,
                ggxg,
                cos_i
            );
            let mut wh = (wi + eta_rel * wo).normalized();
            // normal invert mark
            if wh.z() < 0.0 {
                wh = -wh;
            }

            let partial = ggx_vnpdf_no_d(self.alpha, wi, wh);
            let ndotv = wi * wh;
            let ndotl = wo * wh;

            let sqrt_denom = ndotv + eta_rel * ndotl;
            let eta_rel2 = eta_rel * eta_rel;
            let mut dwh_dwo1 = ndotl / (sqrt_denom * sqrt_denom); // dwh_dwo w/o etas
            let dwh_dwo2 = eta_rel2 * dwh_dwo1; // dwh_dwo w/etas

            // FIXME: determine if this is correct, based on Veach(1998), section 5.2.2.1 (page 143)
            // in radiance mode, the reflectance/transmittance is not scaled by eta^2.
            // in importance_mode, it is scaled by eta^2.
            //
            if transport_mode == TransportMode::Importance {
                dwh_dwo1 = dwh_dwo2;
            }
            debug_assert!(
                wh.0.is_finite().all(),
                "{:?} {:?} {:?} {:?}",
                eta_rel,
                ndotv,
                ndotl,
                sqrt_denom
            );
            let ggxd = ggx_d(self.alpha, wh);
            let weight = ggxd * ggxg * ndotv * dwh_dwo1 / g;
            transmission_pdf = (ggxd * partial * dwh_dwo2).abs();

            let inv_reflectance = 1.0 - self.reflectance(eta_outer, eta_inner, kappa, ndotv);
            transmission = if self.metallic {
                0.0
            } else {
                inv_reflectance * weight.abs()
            };

            debug_assert!(
                !transmission.is_nan(),
                "transmission was nan, self: {:?}, lambda:{:?}, wi:{:?}, wo:{:?}",
                self,
                lambda,
                wi,
                wo
            );
            debug_assert!(
                !transmission_pdf.is_nan(),
                "pdf was nan, self: {:?}, lambda:{:?}, wi:{:?}, wo:{:?}",
                self,
                lambda,
                wi,
                wo
            );
        }

        let refl_prob = self.reflectance_probability(eta_outer, eta_inner, kappa, cos_i);

        let f = glossy + transmission;
        let pdf = refl_prob * glossy_pdf + (1.0 - refl_prob) * transmission_pdf;
        debug_assert!(
            !pdf.is_nan() && !f.is_nan(),
            "{} {} {}",
            refl_prob,
            glossy_pdf,
            transmission_pdf
        );
        (f, pdf.into())
    }
    fn generate_and_evaluate(
        &self,
        lambda: f32,
        _: UV,
        transport_mode: TransportMode,
        sample: Sample2D,
        wi: Vec3,
    ) -> (f32, Option<Vec3>, PDF<f32, SolidAngle>) {
        let eta_inner = self.eta.evaluate_power(lambda);
        let eta_outer = self.eta_o.evaluate_power(lambda);

        debug_assert!(sample.x.is_finite() && sample.y.is_finite(), "{:?}", sample);
        debug_assert!(eta_inner.is_finite(), "{}", lambda);
        // let eta_rel = self.eta_rel(eta_inner, wi);
        // only enable metal effects if permeability is 0
        let kappa = if !self.metallic {
            0.0
        } else {
            self.kappa.evaluate_power(lambda)
        };

        let mut wh = sample_wh(self.alpha, wi, sample).normalized();
        let refl_prob = self.reflectance_probability(eta_outer, eta_inner, kappa, wh * wi);
        debug_assert!(sample.x.is_finite(), "{}", refl_prob);
        debug_assert!(refl_prob.is_finite(), "{} {} {}", eta_inner, kappa, wh * wi);
        let mut did_reflect = false;
        let wo;
        if sample.x <= refl_prob {
            // rescale sample x value to 0 to 1 range
            // sample.x = sample.x / refl_prob;
            // debug_assert!(sample.x.is_finite(), "{}", refl_prob);
            // reflection
            did_reflect = true;
            wo = reflect(wi, wh);
        } else {
            // rescale sample x value to 0 to 1 range
            // sample.x = (sample.x - refl_prob) / (1.0 - refl_prob);
            // transmission

            let eta_rel = 1.0 / self.eta_rel(eta_outer, eta_inner, wi);

            wo = refract(wi, wh, eta_rel).unwrap_or_else(|| {
                did_reflect = true;
                reflect(wi, wh)
            });
        }

        let g = (wi.z() * wo.z()).abs();

        if g == 0.0 {
            return (f32::ZERO, Some(wo), 0.0.into());
        }

        let cos_i;

        let mut glossy = f32::ZERO;
        let mut transmission = f32::ZERO;
        let mut glossy_pdf = 0.0;
        let mut transmission_pdf = 0.0;
        let eta_inner = self.eta.evaluate_power(lambda);
        let eta_outer = self.eta_o.evaluate_power(lambda);
        if did_reflect {
            cos_i = wi * wh;
            let refl = self.reflectance(eta_outer, eta_inner, kappa, cos_i);
            debug_assert!(wh.0.is_finite().all());
            let ggxd = ggx_d(self.alpha, wh);
            let ggxg = ggx_g(self.alpha, wi, wo);
            glossy = refl * (0.25 / g) * ggxd * ggxg;
            if cos_i.abs() == 0.0 {
                glossy_pdf = 0.0;
            } else {
                glossy_pdf = ggx_vnpdf(self.alpha, wi, wh) * 0.25 / cos_i.abs();
            }
            debug_assert!(glossy_pdf.is_finite(), "{:?} {}", self.alpha, cos_i);
            if glossy_pdf == 0.0 {
                trace!(
                    "same hemisphere, {:?} ndv: {} refl: {} d:{} g:{}, g:{} p:{}",
                    wh,
                    cos_i,
                    refl,
                    ggxd,
                    ggxg,
                    glossy,
                    glossy_pdf,
                );
            }
        } else {
            let eta_rel = self.eta_rel(eta_outer, eta_inner, wi);

            let ggxg = ggx_g(self.alpha, wi, wo);
            // let mut wh = (wi + eta_rel * wo).normalized();
            // normal invert mark
            if wh.z() < 0.0 {
                wh = -wh;
            }
            cos_i = wi * wh;
            debug_assert!(
                wi.0.is_finite().all() && wo.0.is_finite().all(),
                "{:?} {:?} {:?} {:?}",
                wi,
                wo,
                ggxg,
                cos_i
            );

            let partial = ggx_vnpdf_no_d(self.alpha, wi, wh);
            let ndotv = wi * wh;
            let ndotl = wo * wh;

            let sqrt_denom = ndotv + eta_rel * ndotl;
            let eta_rel2 = eta_rel * eta_rel;
            let mut dwh_dwo1 = ndotl / (sqrt_denom * sqrt_denom); // dwh_dwo w/o etas
            let dwh_dwo2 = eta_rel2 * dwh_dwo1; // dwh_dwo w/etas

            // in radiance mode, the reflectance/transmittance is not scaled by eta^2.
            // in importance_mode, it is scaled by eta^2.
            if transport_mode == TransportMode::Importance {
                dwh_dwo1 = dwh_dwo2;
            }
            debug_assert!(
                wh.0.is_finite().all(),
                "{:?} {:?} {:?} {:?}",
                eta_rel,
                ndotv,
                ndotl,
                sqrt_denom
            );
            let ggxd = ggx_d(self.alpha, wh);
            let weight = ggxd * ggxg * ndotv * dwh_dwo1 / g;
            transmission_pdf = (ggxd * partial * dwh_dwo2).abs();

            let inv_reflectance = 1.0 - self.reflectance(eta_outer, eta_inner, kappa, ndotv);
            transmission = if self.metallic {
                0.0
            } else {
                inv_reflectance * weight.abs()
            };

            debug_assert!(
                !transmission.is_nan(),
                "transmission was nan, self: {:?}, lambda:{:?}, wi:{:?}, wo:{:?}",
                self,
                lambda,
                wi,
                wo
            );
            debug_assert!(
                !transmission_pdf.is_nan(),
                "pdf was nan, self: {:?}, lambda:{:?}, wi:{:?}, wo:{:?}",
                self,
                lambda,
                wi,
                wo
            );
        }

        let refl_prob = self.reflectance_probability(eta_outer, eta_inner, kappa, cos_i);

        let f = glossy + transmission;
        let pdf = refl_prob * glossy_pdf + (1.0 - refl_prob) * transmission_pdf;
        debug_assert!(
            !pdf.is_nan() && !f.is_nan(),
            "{} {} {}",
            refl_prob,
            glossy_pdf,
            transmission_pdf
        );
        if pdf <= 0.0 {
            trace!(
                "{:?}->{:?}\n {:?} {:?} {:?}",
                wi,
                wo,
                refl_prob,
                glossy_pdf,
                transmission_pdf,
            );
        }
        // generated this direction just now, so pdf should be nonzero and f should be nonzero
        debug_assert!(
            f > 0.0 && pdf > 0.0,
            "{:?}, {:?}, {:?} -> {:?}, wh = {:?}, did_reflect: {}",
            f,
            pdf,
            wi,
            wo,
            wh,
            did_reflect
        );
        (f, Some(wo), pdf.into())
    }

    fn outer_medium_id(&self, _uv: UV) -> MediumId {
        self.outer_medium_id
    }
    fn inner_medium_id(&self, _uv: UV) -> MediumId {
        self.inner_medium_id
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;
    use crate::props::*;

    use math::spectral::BOUNDED_VISIBLE_RANGE;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use super::*;
    use crate::curves;
    use crate::hittable::*;
    use crate::materials::MaterialId;

    #[test]
    fn test_fresnel() {
        let eta_o = 1.004;
        let eta_inner = 1.45;
        // let wi = Vec3::new(0.48507738, 0.4317013, -0.76048267);
        // let wo = Vec3::new(-0.7469567, -0.66481555, 0.00871551);
        let cos_theta_i = -0.76048267;
        let fr_1 = fresnel_dielectric(eta_o, eta_inner, cos_theta_i);
        let fr_2 = fresnel_dielectric(eta_o, eta_inner, -cos_theta_i);
        println!("fr1 is {}, fr2 is {}", fr_1, fr_2);

        let cos_theta_i = 0.00871551;
        let fr_1 = fresnel_dielectric(eta_o, eta_inner, cos_theta_i);
        let fr_2 = fresnel_dielectric(eta_o, eta_inner, -cos_theta_i);
        println!("fr1 is {}, fr2 is {}", fr_1, fr_2);
    }

    fn ggx_glass(roughness: f32) -> GGX {
        let glass = curves::cauchy(1.5, 10000.0);
        let flat_zero = curves::void();
        let flat_one = curves::cie_e(1.0);
        GGX::new(roughness, glass, flat_one, flat_zero, 0, 0)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn test_ggx(roughness in valid_ggx_roughness(), wi in unit_vector(), lambda in 400.0..800.0f32, s in uniform_sample()) {
            let ggx_glass = ggx_glass(roughness);
            let fake_hit_record: HitRecord = HitRecord::new(
                0.0,
                Point3::ZERO,
                UV(0.0f32, 0.0f32),
                lambda,
                Vec3::Z,
                MaterialId::Material(0),
                0,
                None,
            );

            let maybe_wo = ggx_glass.generate(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                s,
                wi,
            );
            assert!(maybe_wo.is_some());

            let wo = maybe_wo.unwrap();
            let (orig_f, orig_pdf) = ggx_glass.bsdf(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                wi,
                wo,
            );

            // check swapping wi and wo
            let (wi, wo) = (wo, wi);
            let (sampled_f, sampled_pdf) = ggx_glass.bsdf(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                wi,
                wo,
            );
            assert!(sampled_f > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
            assert!(*sampled_pdf >= 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",  orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
            assert!(orig_f > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
            assert!(*orig_pdf >= 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);

        }

        #[test]
        fn test_ggx2(roughness in valid_ggx_roughness(), wi in unit_vector(), wo in unit_vector(), lambda in 400.0..800.0f32) {
            let ggx_glass = ggx_glass(roughness);
            let fake_hit_record: HitRecord = HitRecord::new(
                0.0,
                Point3::ZERO,
                UV(0.0f32, 0.0f32),
                lambda,
                Vec3::Z,
                MaterialId::Material(0),
                0,
                None,
            );
            let (orig_f, orig_pdf) = ggx_glass.bsdf(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                wi,
                wo,
            );
            let (wi, wo) = (wo, wi);
            let (sampled_f, sampled_pdf) = ggx_glass.bsdf(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                wi,
                wo,
            );
            assert!(
                sampled_f >= 0.0,
                "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
                orig_f,
                orig_pdf,
                sampled_f,
                sampled_pdf,
                wi,
                wo
            );
            assert!(
                *sampled_pdf >= 0.0,
                "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
                orig_f,
                orig_pdf,
                sampled_f,
                sampled_pdf,
                wi,
                wo
            );
            assert!(
                orig_f >= 0.0,
                "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
                orig_f,
                orig_pdf,
                sampled_f,
                sampled_pdf,
                wi,
                wo
            );
            assert!(
                *orig_pdf >= 0.0,
                "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
                orig_f,
                orig_pdf,
                sampled_f,
                sampled_pdf,
                wi,
                wo
            );
        }
    }

    #[test]
    fn test_ggx_functions() {
        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));

        let ggx_glass = ggx_glass(0.001);
        let lambda = 500.0;
        let fake_hit_record: HitRecord = HitRecord::new(
            0.0,
            Point3::ZERO,
            UV(0.0f32, 0.0f32),
            lambda,
            Vec3::Z,
            MaterialId::Material(0),
            0,
            None,
        );

        let test_many = true;
        let mut wi_s: Vec<Vec3> = Vec::new();

        wi_s.push(Vec3::new(0.01, -0.01, -0.99).normalized());
        wi_s.push(Vec3::new(0.5, -0.1, -0.99).normalized());
        wi_s.push(Vec3::new(0.8, -0.1, -0.49).normalized());

        if test_many {
            for _ in 0..1000000 {
                wi_s.push(random_on_unit_sphere(sampler.draw_2d()));
            }
        }

        let mut succeeded = 0;
        for &wi in wi_s.iter() {
            let maybe_wo = ggx_glass.generate(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                sampler.draw_2d(),
                wi,
            );
            if let Some(wo) = maybe_wo {
                let (orig_f, orig_pdf) = ggx_glass.bsdf(
                    fake_hit_record.lambda,
                    fake_hit_record.uv,
                    fake_hit_record.transport_mode,
                    wi,
                    wo,
                );

                // check swapping wi and wo
                let (wi, wo) = (wo, wi);
                let (sampled_f, sampled_pdf) = ggx_glass.bsdf(
                    fake_hit_record.lambda,
                    fake_hit_record.uv,
                    fake_hit_record.transport_mode,
                    wi,
                    wo,
                );
                assert!(sampled_f > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(*sampled_pdf >= 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",  orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(orig_f > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(*orig_pdf >= 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                succeeded += 1;
            } /* else {
                  print!("x");
              } */
        }
        println!("{} succeeded, {} failed", succeeded, wi_s.len() - succeeded);
        let wi = Vec3::new(0.9709351, 0.18724124, 0.14908342);
        let wo = Vec3::new(-0.008856451, 0.6295874, -0.7768792);

        let (orig_f, orig_pdf) = ggx_glass.bsdf(
            fake_hit_record.lambda,
            fake_hit_record.uv,
            fake_hit_record.transport_mode,
            wi,
            wo,
        );
        let (wi, wo) = (wo, wi);
        let (sampled_f, sampled_pdf) = ggx_glass.bsdf(
            fake_hit_record.lambda,
            fake_hit_record.uv,
            fake_hit_record.transport_mode,
            wi,
            wo,
        );
        assert!(
            sampled_f >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            *sampled_pdf >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            orig_f >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            *orig_pdf >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
    }

    // TODO: debug this failure case
    #[test]
    fn test_failure_case() {
        let lambda = 762.2971;
        let wi = Vec3::new(0.073927574, -0.9872729, 0.1408083);
        let wo = Vec3::new(0.048132252, 0.5836164, -0.81060183);

        let ggx_glass = ggx_glass(0.001);

        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wi, wo);
        println!("{:?} {:?}", f, pdf);
    }
    //wi: Vec3(f32x4(0.48507738, 0.4317013, -0.76048267, -0.0)), wo: Vec3(f32x4(-0.7469567, -0.66481555, 0.00871551, 0.0))
    // TODO: debug this failure case
    #[test]
    fn test_failure_case2() {
        let lambda = 500.0;
        let ggx_glass = ggx_glass(0.01);

        // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
        let wi = Vec3::new(0.48507738, 0.4317013, -0.76048267);
        let wo = Vec3::new(-0.7469567, -0.66481555, 0.00871551);

        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wi, wo);
        println!("normal   {:?} {:?}", f, pdf);
        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wo, wi);
        println!("reversed {:?} {:?}", f, pdf);
    }

    // Vec3(0.95028764, -0.24520797, 0.19190234), wo: Vec3(-0.19736944, 0.961363, -0.19190234)
    // TODO: debug this failure case
    #[test]
    fn test_failure_case3() {
        let lambda = 500.0;
        let ggx_glass = ggx_glass(0.01);

        // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));

        let wi = Vec3::new(0.95028764, -0.24520797, 0.19190234);
        let wo = Vec3::new(-0.19736944, 0.961363, -0.19190234);

        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wi, wo);
        println!("normal   {:?} {:?}", f, pdf);
        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wo, wi);
        println!("reversed {:?} {:?}", f, pdf);
    }
    #[test]
    fn test_extremely_low_roughness() {
        let lambda = 762.2971;
        let wi = Vec3::new(0.073927574, -0.9872729, 0.1408083);
        let wo = Vec3::new(0.048132252, 0.5836164, -0.81060183);
        let ggx_glass = ggx_glass(0.01);

        let (f, pdf) = ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wi, wo);
        println!("{:?} {:?}", f, pdf);
    }

    #[test]
    fn test_integral() {
        let visible_bounds = BOUNDED_VISIBLE_RANGE;
        let ggx_glass = ggx_glass(0.1);

        let n = 10000000;
        let sum: f32 = (0..n)
            .into_par_iter()
            .map(|_| {
                let lambda = visible_bounds.sample(Sample1D::new_random_sample().x);
                let Sample2D { x: u, y: v } = Sample2D::new_random_sample();
                let phi = u * TAU;
                let theta = v * PI;
                let wi = Vec3::new(
                    phi.cos() * theta.cos(),
                    phi.sin() * theta.cos(),
                    theta.sin(),
                );
                let wo = ggx_glass
                    .generate(
                        lambda,
                        UV(0.0, 0.0),
                        TransportMode::Importance,
                        Sample2D::new_random_sample(),
                        wi,
                    )
                    .unwrap();
                let (f, pdf) =
                    ggx_glass.bsdf(lambda, UV(0.0, 0.0), TransportMode::Importance, wi, wo);
                if *pdf == 0.0 {
                    0.0
                } else {
                    wi.z() * f / *pdf
                }
            })
            .sum();
        println!("{}", sum / n as f32);
    }

    #[cfg(feature = "preview")]
    #[test]
    fn test_sampling_direction() {
        use minifb::{Key, WindowOptions};
        use ordered_float::Float;
        use rand::random;

        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        enum Mode {
            ViewGenerate,
            ViewGeneratePDF,
            ViewEval,
            ViewPDF,
        }

        use crate::tonemap::Clamp;

        let mut mat = GGX::new(
            0.001,
            Curve::Cauchy { a: 1.4, b: 30000.0 },
            Curve::Const(1.0),
            Curve::Const(0.0),
            0,
            0,
        );
        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1usize)
        //     .build_global()
        //     .unwrap();

        let (width, height) = (500, 500);
        let mut film = Vec2D::new(width, height, XYZColor::BLACK);

        let mut tonemapper = Clamp::new(0.0, true, true);
        let mut total_samples = 10000;
        let samples_per_iteration = 10;
        let mut incoming_direction = Vec3::Z;
        let mut incoming_direction_zenithal = 0.0; // ranges from 0 to pi
        let mut incoming_direction_azimuthal = 0.0; // ranges from 0 to 2pi
        let mut mode = Mode::ViewGenerate;

        window_loop(
            width,
            height,
            144,
            WindowOptions::default(),
            true,
            |window, mut window_buffer, width, height| {
                let mut reset = false;
                if window.is_key_down(Key::W) {
                    incoming_direction_zenithal =
                        (incoming_direction_zenithal - 0.01).clamp(0.0, PI);
                    reset = true;
                }
                if window.is_key_down(Key::S) {
                    incoming_direction_zenithal =
                        (incoming_direction_zenithal + 0.01).clamp(0.0, PI);
                    reset = true;
                }
                if window.is_key_down(Key::A) {
                    incoming_direction_azimuthal =
                        (incoming_direction_azimuthal - 0.05).clamp(0.0, 2.0 * PI);
                    reset = true;
                }
                if window.is_key_down(Key::D) {
                    incoming_direction_azimuthal =
                        (incoming_direction_azimuthal + 0.05).clamp(0.0, 2.0 * PI);
                    reset = true;
                }

                if window.is_key_down(Key::LeftBracket) {
                    mat.alpha = (mat.alpha * 1.1).clamp(0.0, 1.0);
                    reset = true;
                }
                if window.is_key_down(Key::RightBracket) {
                    mat.alpha = (mat.alpha / 1.1).clamp(0.0, 1.0);
                    reset = true;
                }

                if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
                    mode = match mode {
                        Mode::ViewGenerate => Mode::ViewGeneratePDF,
                        Mode::ViewGeneratePDF => Mode::ViewEval,
                        Mode::ViewEval => Mode::ViewPDF,
                        Mode::ViewPDF => Mode::ViewGenerate,
                    };
                    println!("new mode is now {:?}, resetting film", mode);
                    reset = true;
                }

                if reset {
                    film.buffer.fill(XYZColor::BLACK);
                    incoming_direction = uv_to_direction((
                        incoming_direction_azimuthal / 2.0 / PI,
                        incoming_direction_zenithal / PI,
                    ));
                    total_samples = 0;
                }

                let lambda = EXTENDED_VISIBLE_RANGE.sample(random());
                match mode {
                    Mode::ViewGenerate | Mode::ViewGeneratePDF => {
                        for _ in 0..samples_per_iteration {
                            let (eval, dir, pdf) = mat.generate_and_evaluate(
                                lambda,
                                UV(0.0, 0.0),
                                Default::default(),
                                Sample2D::new_random_sample(),
                                incoming_direction,
                            );
                            let uv = direction_to_uv(dir.unwrap().normalized());
                            let p = (
                                (uv.0.clamp(0.0, 1.0 - f32::EPSILON) * width as f32) as usize,
                                (uv.1.clamp(0.0, 1.0 - f32::EPSILON) * height as f32) as usize,
                            );

                            let jacobian = (uv.1 * PI).sin().abs() * 2.0 * PI * PI;
                            let pixel = film.at(p.0, p.1);
                            let sw = if mode == Mode::ViewGenerate {
                                SingleWavelength::new(lambda, eval)
                            } else {
                                SingleWavelength::new(lambda, *pdf / jacobian)
                            };
                            film.write_at(p.0, p.1, pixel + sw.into());
                        }
                        total_samples += samples_per_iteration;
                    }
                    Mode::ViewEval | Mode::ViewPDF => {
                        film.buffer
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(idx, pixel)| {
                                let (x, y) = (idx % width, idx / width);
                                let (u, v) = (
                                    (x as f32 + 0.5) / width as f32,
                                    (y as f32 + 0.5) / height as f32,
                                );
                                let out_dir = uv_to_direction((u, v));

                                let (eval, pdf) = mat.bsdf(
                                    lambda,
                                    UV(0.0, 0.0),
                                    Default::default(),
                                    incoming_direction,
                                    out_dir,
                                );

                                if (*pdf) == 0.0 {
                                    return;
                                }

                                let sw = if mode == Mode::ViewEval {
                                    SingleWavelength::new(lambda, eval)
                                } else {
                                    SingleWavelength::new(lambda, *pdf)
                                };

                                *pixel += sw.into();
                            });
                        total_samples += 1;
                    }
                }

                let factor = 1.0 / ((total_samples as f32).sqrt() + 1.0);
                update_window_buffer(&mut window_buffer, &film, &mut tonemapper, factor);
            },
        );
    }
}
