use crate::hittable::HitRecord;
use crate::material::{Material, BRDF, PDF};
use crate::math::*;

pub fn reflect(wi: Vec3, normal: Vec3) -> Vec3 {
    let wi = -wi;
    wi - 2.0 * (wi * normal) * normal
}

pub fn fresnel_dielectric(eta_i: f32, eta_t: f32, cos_i: f32) -> f32 {
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

pub fn fresnel_conductor(
    mut eta_i: f32,
    mut eta_t: f32,
    mut k_t: f32,
    mut cos_theta_i: f32,
) -> f32 {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    // handle dielectrics
    let swapped = cos_theta_i < 0.0;

    cos_theta_i = if swapped { -cos_theta_i } else { cos_theta_i };

    let new_eta_i = if swapped { eta_t } else { eta_i };
    let new_eta_t = if swapped { eta_i } else { eta_t };

    // set k_t to zero when dielectric
    // k_t = if swapped { 0.0 } else { k_t };

    eta_i = new_eta_i;
    eta_t = new_eta_t;

    // onto the full equations

    let eta = eta_t / eta_i;
    let etak = k_t / eta_i;

    let cos_theta_i2 = cos_theta_i * cos_theta_i;
    let sin_theta_i2 = 1.0 - cos_theta_i2;
    let eta2 = eta * eta;
    let etak2 = etak * etak;

    let t0 = eta2 - etak2 - sin_theta_i2;
    assert!(t0 * t0 + eta2 * etak2 > 0.0);
    let a2plusb2 = (t0 * t0 + eta2 * etak2 * 4.0).sqrt();
    let t1 = a2plusb2 + cos_theta_i2;
    assert!(a2plusb2 + t0 > 0.0);
    let a = ((a2plusb2 + t0) * 0.5).sqrt();
    let t2 = a * cos_theta_i * 2.0;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = a2plusb2 * cos_theta_i2 + sin_theta_i2 * sin_theta_i2;
    let t4 = t2 * sin_theta_i2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    (rs * rs + rp * rp) / 2.0
}

pub fn refract(wi: Vec3, normal: Vec3, ior: f32) -> Option<Vec3> {
    let mut cos_i = wi * normal;
    let mut normal = normal;
    let mut eta_i = 1.0;
    let mut eta_t = ior;
    if (cos_i < 0.0) {
        cos_i = -cos_i;
    } else {
        normal = -normal;
        let (eta_i, eta_t) = (eta_t, eta_i);
    }
    let eta = eta_i / eta_t;
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        Some(eta * wi + (eta * cos_i - k.sqrt()) * normal)
    }
}

fn ggx_d(alpha: f32, wm: Vec3) -> f32 {
    let slope = (wm.x() / alpha, wm.y() / alpha);
    let slope2 = (slope.0 * slope.0, slope.1 * slope.1);
    let t = wm.z() * wm.z() + slope2.0 + slope2.1;
    1.0 / (PI * alpha * alpha * t * t)
}

fn ggx_lambda(alpha: f32, w: Vec3) -> f32 {
    if w.z() == 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let w2 = Vec3::from_raw(w.0 * w.0);
    let c = 1.0 + (a2 * w2.x() + a2 * w2.y()) / w2.z(); // replace a2 with Vec2 for anistropy
    c.sqrt() * 0.5 - 0.5
}

fn ggx_g(alpha: f32, wi: Vec3, wo: Vec3) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, wi) + ggx_lambda(alpha, wo))
}

fn ggx_vnpdf(alpha: f32, wi: Vec3, wh: Vec3) -> f32 {
    let inv_gl = 1.0 + ggx_lambda(alpha, wi);
    (ggx_d(alpha, wh) * (wi * wh).abs()) / (inv_gl * wi.z().abs())
}

fn ggx_vnpdf_no_d(alpha: f32, wi: Vec3, wh: Vec3) -> f32 {
    ((wi * wh) / ((1.0 + ggx_lambda(alpha, wi)) * wi.z())).abs()
}

fn ggx_pdf(alpha: f32, wi: Vec3, wh: Vec3) -> f32 {
    ggx_d(alpha, wh) * wh.z().abs()
}

fn sample_vndf(alpha: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
    let Sample2D { x, y } = sample;
    let v = Vec3::new(alpha * wi.x(), alpha * wi.y(), wi.z()).normalized();

    let t1 = if v.z() < 0.9999 {
        v.cross(Vec3::Z).normalized()
    } else {
        Vec3::X
    };
    let t2 = t1.cross(v);
    let a = 1.0 / (1.0 + v.z());
    let r = x.sqrt();
    let phi = if y < a {
        y / a * PI
    } else {
        PI + (y - a) / (1.0 - a) * PI
    };

    let (sin_phi, cos_phi) = phi.sin_cos();
    let p1 = r * cos_phi;
    let p2 = r * sin_phi * if y < a { 1.0 } else { v.z() };
    let n = p1 * t1 + p2 * t2 + (1.0 - p1 * p1 - p2 * p2).sqrt() * v;

    Vec3::new(alpha * n.x(), alpha * n.y(), n.z().max(0.0)).normalized()
}

fn sample_wh(alpha: f32, wi: Vec3, sampler: &mut Box<dyn Sampler>) -> Vec3 {
    let sample = sampler.draw_2d();
    let flip = wi.z() < 0.0;
    let wh = sample_vndf(alpha, if flip { -wi } else { wi }, sample);
    if flip {
        -wh
    } else {
        wh
    }
}

#[derive(Debug)]
pub struct GGX {
    roughness: f32,
    eta: SPD,
    eta_o: f32,
    kappa: SPD,
    permeability: f32,
}

impl GGX {
    pub fn new(roughness: f32, eta: SPD, eta_o: f32, kappa: SPD, permeability: f32) -> Self {
        GGX {
            roughness,
            eta,
            eta_o,
            kappa,
            permeability,
        }
    }

    fn reflectance(&self, eta_inner: f32, kappa: f32, cos_theta_i: f32) -> f32 {
        if self.permeability > 0.0 {
            fresnel_dielectric(self.eta_o, eta_inner, cos_theta_i)
        } else {
            fresnel_conductor(self.eta_o, eta_inner, kappa, cos_theta_i)
        }
    }

    fn reflectance_probability(&self, eta_inner: f32, kappa: f32, cos_theta_i: f32) -> f32 {
        if self.permeability > 0.0 {
            // fresnel_dielectric(self.eta_o, eta_i, wi.z())
            // scale by self.permeability
            (self.permeability * self.reflectance(eta_inner, kappa, cos_theta_i) + 1.0
                - self.permeability)
                .clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

impl PDF for GGX {
    fn value(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> f32 {
        let cos_i = wi.z();
        let same_hemisphere = cos_i * wo.z() > 0.0;
        let outside: bool = cos_i >= 0.0;

        let g = (wi.z() * wo.z()).abs();

        let mut glossy_pdf = 0.0;
        let mut transmission_pdf = 0.0;
        let eta_inner = self.eta.evaluate_power(hit.lambda);
        let kappa = if self.permeability > 0.0 {
            0.0
        } else {
            self.kappa.evaluate_power(hit.lambda)
        };
        if same_hemisphere {
            let mut wh = (wo + wi);
            if wh.z() < 0.0 {
                wh = -wh;
            }
            let ndotv = wi * wh;
            let refl = self.reflectance(eta_inner, kappa, cos_i);
            glossy_pdf = ggx_vnpdf(self.roughness, wi, wh) * 0.25 / ndotv.abs();
        } else {
            if self.permeability > 0.0 {
                let mut eta_rel = self.eta_o / eta_inner;
                if wi.z() < 0.0 {
                    eta_rel = 1.0 / eta_rel;
                }

                let mut wh = (wi + eta_rel * wo);
                if wh.z() < 0.0 {
                    wh = -wh;
                }

                let ggxg = ggx_g(self.roughness, wi, wo);
                let partial = ggx_vnpdf_no_d(self.roughness, wi, wh);
                let mut cos_theta_i = 0.0;
                let ndotv = wi * wh;
                let ndotl = wo * wh;

                let sqrt_denom = ndotv + eta_rel * ndotl;
                if sqrt_denom.abs() < 1e-6 {
                    transmission_pdf = 0.0;
                } else {
                    let dwh_dwo = (eta_rel * eta_rel * ndotl) / (sqrt_denom * sqrt_denom);
                    let ggxd = ggx_d(self.roughness, wh);
                    // let weight = ggxd * ggxg * ndotv * dwh_dwo / g;
                    // transmission.0 = weight;
                    cos_theta_i = ndotv;

                    // let inv_reflectance = 1.0 - self.reflectance(eta_i, kappa, cos_theta_i);
                    transmission_pdf =
                        (ggxd * ggx_vnpdf_no_d(self.roughness, wi, wh) * dwh_dwo).abs();
                }
            }
        }

        let refl_prob = self.reflectance_probability(eta_inner, kappa, cos_i);

        refl_prob * glossy_pdf + (1.0 - refl_prob) * transmission_pdf
    }
    fn generate(
        &self,
        hit: &HitRecord,
        mut sampler: &mut Box<dyn Sampler>,
        wi: Vec3,
    ) -> Option<Vec3> {
        let eta_inner = self.eta.evaluate_power(hit.lambda);
        let kappa = if self.permeability > 0.0 {
            0.0
        } else {
            self.kappa.evaluate_power(hit.lambda)
        };
        let refl_prob = self.reflectance_probability(eta_inner, kappa, wi.z());
        if refl_prob == 1.0 || sampler.draw_1d().x < refl_prob {
            // reflection
            let wh = sample_wh(self.roughness, wi, &mut sampler);
            let wo = reflect(wi, wh);
            return Some(wo);
        } else {
            // transmission
            let mut wh = sample_wh(self.roughness, wi, &mut sampler);
            let mut eta_rel = self.eta_o / eta_inner;
            if wi.z() < 0.0 {
                eta_rel = 1.0 / eta_rel;
            }

            if wh.z() < 0.0 {
                wh = -wh;
            }
            let wo = refract(wi, wh, eta_rel);
            return wo;
        }
    }
}

impl BRDF for GGX {
    fn f(&self, hit: &HitRecord, wi: Vec3, wo: Vec3) -> SingleEnergy {
        let cos_i = wi.z();
        let same_hemisphere = cos_i * wo.z() > 0.0;
        let outside: bool = cos_i >= 0.0;

        let g = (wi.z() * wo.z()).abs();

        let mut glossy = SingleEnergy::ZERO;
        let mut transmission = SingleEnergy::ZERO;
        let eta_inner = self.eta.evaluate_power(hit.lambda);
        let kappa = if self.permeability > 0.0 {
            0.0
        } else {
            self.kappa.evaluate_power(hit.lambda)
        };
        if same_hemisphere {
            let mut wh = (wo + wi);
            if wh.z() < 0.0 {
                wh = -wh;
            }
            let ndotv = wi * wh;
            let refl = self.reflectance(eta_inner, kappa, cos_i);
            glossy.0 =
                refl * (0.25 / g) * ggx_d(self.roughness, wh) * ggx_g(self.roughness, wi, wo);
        } else {
            if self.permeability > 0.0 {
                let mut eta_rel = self.eta_o / eta_inner;
                if wi.z() < 0.0 {
                    eta_rel = 1.0 / eta_rel;
                }

                let mut wh = (wi + eta_rel * wo);
                if wh.z() < 0.0 {
                    wh = -wh;
                }

                let ggxg = ggx_g(self.roughness, wi, wo);
                let partial = ggx_vnpdf_no_d(self.roughness, wi, wh);
                let mut cos_theta_i = 0.0;
                let ndotv = wi * wh;
                let ndotl = wo * wh;

                let sqrt_denom = ndotv + eta_rel * ndotl;
                let dwh_dwo = (eta_rel * eta_rel * ndotl) / (sqrt_denom * sqrt_denom);
                let ggxd = ggx_d(self.roughness, wh);
                let weight = ggxd * ggxg * ndotv * dwh_dwo / g;
                // transmission.0 = weight;
                cos_theta_i = ndotv;

                let inv_reflectance = 1.0 - self.reflectance(eta_inner, kappa, cos_theta_i);
                transmission.0 = self.permeability * inv_reflectance * weight.abs();
            }
        }

        glossy + transmission
    }
    fn emission(&self, hit: &HitRecord, wi: Vec3, wo: Option<Vec3>) -> SingleEnergy {
        SingleEnergy::ZERO
    }
}

impl Material for GGX {}
