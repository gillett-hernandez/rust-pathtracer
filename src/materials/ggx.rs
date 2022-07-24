use crate::materials::Material;
use crate::math::*;
use crate::world::TransportMode;

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
    pub alpha: f32,
    pub eta: Curve,
    pub eta_o: f32, // replace with SPD
    pub kappa: Curve,
    pub permeability: f32,
    pub outer_medium_id: usize,
}

impl GGX {
    pub fn new(
        roughness: f32,
        eta: Curve,
        eta_o: f32,
        kappa: Curve,
        permeability: f32,
        outer_medium_id: usize,
    ) -> Self {
        debug_assert!(roughness > 0.0);
        GGX {
            alpha: roughness,
            eta,
            eta_o,
            kappa,
            permeability,
            outer_medium_id,
        }
    }

    fn reflectance(&self, eta_inner: f32, kappa: f32, cos_theta_i: f32) -> f32 {
        if self.permeability > 0.0 {
            fresnel_dielectric(self.eta_o, eta_inner, cos_theta_i)
        // if cos_theta_i >= 0.0 {
        // fresnel_dielectric(eta_inner, self.eta_o, cos_theta_i)
        // } else {
        // }
        } else {
            fresnel_conductor(self.eta_o, eta_inner, kappa, cos_theta_i)
            // fresnel_conductor(eta_inner, self.eta_o, kappa, cos_theta_i)
            // if cos_theta_i >= 0.0 {
            // } else {
            // }
        }
    }

    fn reflectance_probability(&self, eta_inner: f32, kappa: f32, cos_theta_i: f32) -> f32 {
        if self.permeability > 0.0 {
            // fresnel_dielectric(self.eta_o, eta_i, wi.z())
            (self.permeability * self.reflectance(eta_inner, kappa, cos_theta_i) + 1.0
                - self.permeability)
                .clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
    fn eta_rel(&self, eta_inner: f32, wi: Vec3) -> f32 {
        if wi.z() < 0.0 {
            self.eta_o / eta_inner
        } else {
            eta_inner / self.eta_o
        }
    }
}

impl GGX {
    fn eval_pdf(
        &self,
        lambda: f32,
        wi: Vec3,
        wo: Vec3,
        transport_mode: TransportMode,
    ) -> (SingleEnergy, PDF) {
        let wi = wi.normalized();
        let same_hemisphere = wi.z() * wo.z() > 0.0;

        let g = (wi.z() * wo.z()).abs();

        if g == 0.0 {
            return (SingleEnergy::ZERO, 0.0.into());
        }

        let cos_i = wi.z();

        let mut glossy = SingleEnergy::ZERO;
        let mut transmission = SingleEnergy::ZERO;
        let mut glossy_pdf = 0.0;
        let mut transmission_pdf = 0.0;
        let eta_inner = self.eta.evaluate_power(lambda);
        let kappa = if self.permeability > 0.0 {
            0.0
        } else {
            self.kappa.evaluate_power(lambda)
        };
        if same_hemisphere {
            let mut wh = (wo + wi).normalized();
            // normal invert mark
            if wh.z() < 0.0 {
                wh = -wh;
            }

            let ndotv = wi * wh;
            let refl = self.reflectance(eta_inner, kappa, ndotv);
            debug_assert!(wh.0.is_finite().all());
            glossy.0 = refl * (0.25 / g) * ggx_d(self.alpha, wh) * ggx_g(self.alpha, wi, wo);
            if ndotv.abs() == 0.0 {
                glossy_pdf = 0.0;
            } else {
                glossy_pdf = ggx_vnpdf(self.alpha, wi, wh) * 0.25 / ndotv.abs();
            }
            debug_assert!(glossy_pdf.is_finite(), "{:?} {}", self.alpha, ndotv);
        } else if self.permeability > 0.0 {
            let eta_rel = self.eta_rel(eta_inner, wi);

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
            match transport_mode {
                // in radiance mode, the reflectance/transmittance is not scaled by eta^2.
                // in importance_mode, it is scaled by eta^2.
                TransportMode::Importance => dwh_dwo1 = dwh_dwo2,
                _ => {}
            };
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

            let inv_reflectance = 1.0 - self.reflectance(eta_inner, kappa, ndotv);
            transmission.0 = self.permeability * inv_reflectance * weight.abs();
            // println!("{:?}, {:?}, {:?}", eta_inner, kappa, ndotv);
            // println!(
            //     "transmission = {:?} = {:?}*{:?}*{:?}*{:?}*{:?}/{:?}",
            //     transmission, inv_reflectance, ggxd, ggxg, ndotv, dwh_dwo1, g
            // );

            // println!(
            //     "transmission_pdf = {:?} = {:?}*{:?}*{:?}",
            //     transmission_pdf, ggxd, partial, dwh_dwo1
            // );

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

        let refl_prob = self.reflectance_probability(eta_inner, kappa, cos_i);
        // println!("glossy: {:?}, transmission: {:?}", glossy, transmission);
        // println!(
        //     "glossy_pdf: {:?}, transmission_pdf: {:?}, refl_prob: {:?}",
        //     glossy_pdf, transmission_pdf, refl_prob
        // );
        // println!("cos_i: {:?}", cos_i);
        // println!();

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
    pub const NAME: &'static str = "GGX";
}

impl Material for GGX {
    fn generate(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        _transport_mode: TransportMode,
        sample: Sample2D,
        wi: Vec3,
    ) -> Option<Vec3> {
        // debug_assert!(sample.x.is_finite() && sample.y.is_finite(), "{:?}", sample);
        // let eta_inner = self.eta.evaluate_power(lambda);
        // debug_assert!(eta_inner.is_finite(), "{}", lambda);
        // // let eta_rel = self.eta_rel(eta_inner, wi);
        // let kappa = if self.permeability > 0.0 {
        //     0.0
        // } else {
        //     self.kappa.evaluate_power(lambda)
        // };
        // let refl_prob = self.reflectance_probability(eta_inner, kappa, wi.z());
        // debug_assert!(refl_prob.is_finite(), "{} {} {}", eta_inner, kappa, wi.z());
        // if refl_prob == 1.0 || sample.x < refl_prob {
        //     // rescale sample x value to 0 to 1 range
        //     sample.x = sample.x / refl_prob;
        //     debug_assert!(sample.x.is_finite(), "{}", refl_prob);
        //     // reflection
        //     let wh = sample_wh(self.alpha, wi, sample).normalized();
        //     let wo = reflect(wi, wh);
        //     return Some(wo);
        // } else {
        //     // rescale sample x value to 0 to 1 range
        //     sample.x = (sample.x - refl_prob) / (1.0 - refl_prob);
        //     debug_assert!(sample.x.is_finite(), "{}", refl_prob);
        //     // transmission
        //     let wh = sample_wh(self.alpha, wi, sample).normalized();

        //     let eta_rel = 1.0 / self.eta_rel(eta_inner, wi);

        //     let wo = refract(wi, wh, eta_rel);
        //     /* if wo.is_none() {
        //         println!("wo was none, because refract returned none (should have been total internal reflection but fresnel was {} and eta_rel was {})", refl_prob, eta_rel);
        //         //     wo = Some(reflect(wi, -wh));
        //     } */
        //     return wo;
        // }
        debug_assert!(sample.x.is_finite() && sample.y.is_finite(), "{:?}", sample);
        let eta_inner = self.eta.evaluate_power(lambda);
        debug_assert!(eta_inner.is_finite(), "{}", lambda);
        // let eta_rel = self.eta_rel(eta_inner, wi);
        // only enable metal effects if permeability is 0
        let kappa = if self.permeability > 0.0 {
            0.0
        } else {
            self.kappa.evaluate_power(lambda)
        };
        let wh = sample_wh(self.alpha, wi, sample).normalized();
        let refl_prob = self.reflectance_probability(eta_inner, kappa, wh * wi);
        debug_assert!(sample.x.is_finite(), "{}", refl_prob);
        debug_assert!(refl_prob.is_finite(), "{} {} {}", eta_inner, kappa, wh * wi);
        if refl_prob == 1.0 || sample.x < refl_prob {
            // rescale sample x value to 0 to 1 range
            // sample.x = sample.x / refl_prob;
            // debug_assert!(sample.x.is_finite(), "{}", refl_prob);
            // reflection
            let wo = reflect(wi, wh);
            return Some(wo);
        } else {
            // rescale sample x value to 0 to 1 range
            // sample.x = (sample.x - refl_prob) / (1.0 - refl_prob);
            // transmission

            let eta_rel = 1.0 / self.eta_rel(eta_inner, wi);

            let mut wo = refract(wi, wh, eta_rel);
            if wo.is_none() {
                // println!("wo was none, because refract returned none (should have been total internal reflection but fresnel was {} and eta_rel was {})", refl_prob, eta_rel);
                wo = Some(reflect(wi, wh));
            }
            return wo;
        }
    }
    fn bsdf(
        &self,
        lambda: f32,
        _uv: (f32, f32),
        transport_mode: TransportMode,
        wi: Vec3,
        wo: Vec3,
    ) -> (SingleEnergy, PDF) {
        self.eval_pdf(lambda, wi, wo, transport_mode)
    }
    fn inner_medium_id(&self, _uv: (f32, f32)) -> usize {
        0
    }
    fn outer_medium_id(&self, _uv: (f32, f32)) -> usize {
        self.outer_medium_id
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use spectral::BOUNDED_VISIBLE_RANGE;

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

    #[test]
    fn test_ggx_functions() {
        let glass = curves::cauchy(1.5, 10000.0);
        let flat_zero = curves::void();
        let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0, 0);

        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));

        let lambda = 500.0;
        let fake_hit_record: HitRecord = HitRecord::new(
            0.0,
            Point3::ZERO,
            (0.0f32, 0.0f32),
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
            for _ in 0..100000 {
                wi_s.push(random_on_unit_sphere(sampler.draw_2d()));
            }
        }

        let mut succeeded = 0;
        for wi in wi_s.iter() {
            let maybe_wo = ggx_glass.generate(
                fake_hit_record.lambda,
                fake_hit_record.uv,
                fake_hit_record.transport_mode,
                sampler.draw_2d(),
                *wi,
            );
            if let Some(wo) = maybe_wo {
                let (orig_f, orig_pdf) = ggx_glass.bsdf(
                    fake_hit_record.lambda,
                    fake_hit_record.uv,
                    fake_hit_record.transport_mode,
                    *wi,
                    wo,
                );

                // check swapping wi and wo
                let (wi, wo) = (wo, wi);
                let (sampled_f, sampled_pdf) = ggx_glass.bsdf(
                    fake_hit_record.lambda,
                    fake_hit_record.uv,
                    fake_hit_record.transport_mode,
                    wi,
                    *wo,
                );
                assert!(sampled_f.0 > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(sampled_pdf.0 >= 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",  orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(orig_f.0 > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
                assert!(orig_pdf.0 > 0.0, "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}", orig_f, orig_pdf, sampled_f, sampled_pdf, wi, wo);
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
            sampled_f.0 >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            sampled_pdf.0 >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            orig_f.0 >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
        assert!(
            orig_pdf.0 >= 0.0,
            "original f: {:?}, pdf: {:?}, swapped f: {:?}, pdf: {:?}, swapped wi: {:?}, wo: {:?}",
            orig_f,
            orig_pdf,
            sampled_f,
            sampled_pdf,
            wi,
            wo
        );
    }

    #[test]
    fn test_failure_case() {
        let lambda = 762.2971;
        let wi = Vec3::new(0.073927574, -0.9872729, 0.1408083);
        let wo = Vec3::new(0.048132252, 0.5836164, -0.81060183);

        let glass = curves::cauchy(1.45, 3540.0);
        let flat_zero = curves::void();
        let ggx_glass = GGX::new(0.01, glass, 1.0, flat_zero, 1.0, 0);

        let (f, pdf) = ggx_glass.eval_pdf(lambda, wi, wo, TransportMode::Importance);
        println!("{:?} {:?}", f, pdf);
    }
    //wi: Vec3(f32x4(0.48507738, 0.4317013, -0.76048267, -0.0)), wo: Vec3(f32x4(-0.7469567, -0.66481555, 0.00871551, 0.0))
    #[test]
    fn test_failure_case2() {
        let lambda = 500.0;

        let glass = curves::cauchy(1.5, 10000.0);
        let flat_zero = curves::void();
        let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0, 0);

        // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
        let wi = Vec3::new(0.48507738, 0.4317013, -0.76048267);
        let wo = Vec3::new(-0.7469567, -0.66481555, 0.00871551);

        let (f, pdf) = ggx_glass.eval_pdf(lambda, wi, wo, TransportMode::Importance);
        println!("{:?} {:?}", f, pdf);
        let (f, pdf) = ggx_glass.eval_pdf(lambda, wo, wi, TransportMode::Importance);
        println!("{:?} {:?}", f, pdf);
    }
    #[test]
    fn test_extremely_low_roughness() {
        let lambda = 762.2971;
        let wi = Vec3::new(0.073927574, -0.9872729, 0.1408083);
        let wo = Vec3::new(0.048132252, 0.5836164, -0.81060183);

        let glass = curves::cauchy(1.45, 3540.0);
        let flat_zero = curves::void();
        let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0, 0);

        let (f, pdf) = ggx_glass.eval_pdf(lambda, wi, wo, TransportMode::Importance);
        println!("{:?} {:?}", f, pdf);
    }

    #[test]
    fn test_integral() {
        let visible_bounds = BOUNDED_VISIBLE_RANGE;
        let glass = curves::cauchy(1.45, 3540.0);
        let flat_zero = curves::void();
        let ggx_glass = GGX::new(0.1, glass, 1.0, flat_zero, 1.0, 0);
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
                        (0.0, 0.0),
                        TransportMode::Importance,
                        Sample2D::new_random_sample(),
                        wi,
                    )
                    .unwrap();
                let (f, pdf) = ggx_glass.eval_pdf(lambda, wi, wo, TransportMode::Importance);
                if pdf.0 == 0.0 {
                    0.0
                } else {
                    wi.z() * f.0 / pdf.0
                }
            })
            .sum();
        println!("{}", sum / n as f32);
    }
}
