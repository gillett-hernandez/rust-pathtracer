use crate::lens_assembly::*;
use crate::lens_interface::*;

use math::*;
// use rand::prelude::*;

pub fn spectrum_cauchy_from_abbe_num(nd: f32, vd: f32) -> (f32, f32) {
    if vd == 0.0 {
        (nd, 0.0)
    } else {
        const LC: f32 = 0.6563;
        const LF: f32 = 0.4861;
        const LD: f32 = 0.587561;
        const LC2: f32 = LC * LC;
        const LF2: f32 = LF * LF;
        const C: f32 = LC2 * LF2 / (LC2 - LF2);
        let b = (nd - 1.0) / vd * C;
        (nd - b / (LD * LD), b)
    }
}

pub fn spectrum_eta_from_abbe_num(nd: f32, vd: f32, lambda: f32) -> f32 {
    let (a, b) = spectrum_cauchy_from_abbe_num(nd, vd);
    a + b / (lambda * lambda)
}

const INTENSITY_EPS: f32 = 0.0001;

pub fn trace_spherical(
    ray: Ray,
    r: f32,
    center: f32,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i16> {
    let scv = Vec3::from(ray.origin - Vec3::Z * center);
    let a = ray.direction * ray.direction;
    let b = 2.0 * ray.direction * scv;
    let c = scv * scv - r * r;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        Err(4)
    } else {
        let mut error = 0;
        let t;
        let a2 = 2.0 * a;
        let t0 = (-b - discriminant.sqrt()) / a2;
        let t1 = (-b + discriminant.sqrt()) / a2;
        if t0 < -1.0e-4 {
            t = t1;
        } else {
            t = t0.min(t1);
        }
        if t < -1.0e-4 {
            Err(16)
        } else {
            let ray = ray.at_time(t);
            let (rx, ry) = (ray.origin.x(), ray.origin.y());
            error |= (rx * rx + ry * ry > housing_radius * housing_radius) as i16;
            let normal = Vec3::new(rx, ry, ray.origin.z() - center) / r;
            if error == 0 {
                Ok((ray, normal.normalized()))
            } else {
                Err(error)
            }
        }
    }
}

pub fn evaluate_aspherical(pos: Point3, r: f32, k: i32, correction: f32x4) -> f32 {
    let h = (pos.x() * pos.x() + pos.y() * pos.y()).sqrt();
    let hr = h / r;
    let h2 = h * h;
    let h4 = h2 * h2;
    let h6 = h4 * h2;
    let h8 = h4 * h4;
    let h10 = h8 * h2;
    let corv = f32x4::new(h4, h6, h8, h10);
    h * hr / (1.0 + (1.0 - (1.0 + k as f32) * hr * hr).max(0.0).sqrt()) + (correction * corv).sum()
}

pub fn evaluate_aspherical_derivative(pos: Point3, r: f32, k: i32, correction: f32x4) -> f32 {
    let h = (pos.x() * pos.x() + pos.y() * pos.y()).sqrt();
    let hr = h / r;
    let h2 = h * h;
    let h3 = h2 * h;

    let h4 = h2 * h2;
    let h5 = h3 * h2;
    let h6 = h4 * h2;
    let h7 = h4 * h3;
    let h9 = h6 * h3;
    let corv = f32x4::new(4.0 * h3, 6.0 * h5, 8.0 * h7, 10.0 * h9);
    let hr2 = hr * hr;
    let subexpr = (1.0 - (1.0 + k as f32) * hr2).max(0.0).sqrt();
    2.0 * hr / (1.0 + subexpr)
        + hr2 * hr * (k as f32 + 1.0) / (subexpr * (subexpr + 1.0).powf(2.0))
        + (correction * corv).sum()
}

pub fn trace_aspherical(
    mut ray: Ray,
    r: f32,
    center: f32,
    k: i32,
    mut correction: f32x4,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i32> {
    let mut t = 0.0;
    let result = trace_spherical(ray, r, center, housing_radius)?;
    ray = result.0;
    let normal = result.1;
    let mut rad = r;
    if (center + r - ray.origin.z()).abs() > (center - r - ray.origin.z()).abs() {
        rad = -r;
        correction = -correction;
    }

    let mut position_error;
    // repeatedly trace the ray forwads and backwards until the position error is less than some constant.
    for _ in 0..100 {
        position_error =
            rad + center - ray.origin.z() - evaluate_aspherical(ray.origin, rad, k, correction);
        let terr = position_error / ray.direction.z();
        t += terr;
        ray = ray.at_time(terr);
        if position_error.abs() < 1.0e-4 {
            break;
        }
    }
    let dz = evaluate_aspherical_derivative(ray.origin, rad, k, correction)
        * if normal.z() < 0.0 { -1.0 } else { 1.0 };
    let sqr = ray.origin.0 * ray.origin.0;
    let new_r = (sqr.extract(0) + sqr.extract(1)).sqrt();
    let normal = Vec3::new(
        ray.origin.x() / new_r * dz,
        ray.origin.y() / new_r * dz,
        normal.z() / normal.z().abs(),
    )
    .normalized();

    Ok((ray.at_time(t), normal))
}

pub fn trace_cylindrical(
    mut ray: Ray,
    r: f32,
    center: f32,
    housing_radius: f32,
) -> Result<(Ray, Vec3), i32> {
    let scv = Vec3::new(ray.origin.x(), 0.0, ray.origin.z() - center);
    let a = ray.direction * ray.direction;
    let b = 2.0 * ray.direction * scv;
    let c = scv * scv - r * r;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Err(4);
    }
    let t;
    if r > 0.0 {
        t = (-b - discriminant.sqrt()) / (2.0 * a);
    } else {
        t = (-b + discriminant.sqrt()) / (2.0 * a);
    }
    ray = ray.at_time(t);
    let sqr = ray.origin.0 * ray.origin.0;
    if sqr.extract(0) + sqr.extract(1) > housing_radius * housing_radius {
        return Err(8);
    }
    let normal = Vec3::new(ray.origin.x(), 0.0, ray.origin.z() - center) / r;
    Ok((ray, normal))
}

pub fn fresnel(n1: f32, n2: f32, cosr: f32, cost: f32) -> f32 {
    if cost <= 0.0 {
        1.0
    } else {
        let n2cost = n2 * cost;
        let n1cosr = n1 * cosr;
        let n1cost = n1 * cost;
        let n2cosr = n2 * cosr;
        let rs = (n1cosr - n2cost) / (n1cosr + n2cost);
        let rp = (n1cost - n2cosr) / (n1cost + n2cosr);
        ((rs * rs + rp * rp) / 2.0).min(1.0)
    }
}

pub fn refract(n1: f32, n2: f32, normal: Vec3, dir: Vec3) -> (Vec3, f32) {
    if n1 == n2 {
        (dir, 1.0)
    } else {
        let eta = n1 / n2;
        let norm = dir.norm();
        let cos1 = -(normal * dir) / norm;
        let cos2_2 = 1.0 - eta * eta * (1.0 - cos1 * cos1);
        if cos2_2 < 0.0 {
            (dir, 0.0)
        } else {
            let cos2 = cos2_2.sqrt();
            (
                dir * eta / norm + (eta * cos1 - cos2) * normal,
                1.0 - fresnel(n1, n2, cos1, cos2),
            )
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Input<T> {
    pub ray: T,
    pub lambda: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Output<T> {
    pub ray: T,
    pub tau: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct PlaneRay(pub f32x4);

impl PlaneRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SphereRay(pub f32x4);

impl SphereRay {
    pub fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            0: f32x4::new(x, y, dx, dy),
        }
    }
    pub fn x(&self) -> f32 {
        self.0.extract(0)
    }
    pub fn y(&self) -> f32 {
        self.0.extract(1)
    }
    pub fn dx(&self) -> f32 {
        self.0.extract(2)
    }
    pub fn dy(&self) -> f32 {
        self.0.extract(3)
    }
}

impl From<SphereRay> for PlaneRay {
    fn from(other: SphereRay) -> Self {
        Self { 0: other.0 }
    }
}

impl From<PlaneRay> for SphereRay {
    fn from(other: PlaneRay) -> Self {
        Self { 0: other.0 }
    }
}

pub fn plane_to_camera_space(ray_in: PlaneRay, plane_pos: f32) -> Ray {
    let [x, y, dx, dy]: [f32; 4] = ray_in.0.into();
    Ray::new(
        Point3::new(x, y, plane_pos),
        Vec3::new(dx, dy, 1.0).normalized(),
    )
}

pub fn camera_space_to_plane(ray_in: Ray, plane_pos: f32) -> PlaneRay {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let [dx, dy, dz, _]: [f32; 4] = ray_in.direction.0.into();
    let t = (plane_pos - z) / dz;

    PlaneRay::new(x + t * dx, y + t * dy, dx / dz.abs(), dy / dz.abs())
}

pub fn sphere_to_camera_space(ray_in: SphereRay, sphere_center: f32, sphere_radius: f32) -> Ray {
    let [x, y, dx, dy]: [f32; 4] = ray_in.0.into();
    let normal = Vec3::new(
        x / sphere_radius,
        y / sphere_radius,
        (sphere_radius * sphere_radius - x * x - y * y)
            .max(0.0)
            .sqrt()
            / sphere_radius.abs(),
    );
    let temp_direction = Vec3::new(dx, dy, (1.0 - dx * dx - dy * dy).max(0.0).sqrt());
    let ex = Vec3::new(normal.z(), 0.0, -normal.x()).normalized();
    let frame = TangentFrame::from_tangent_and_normal(ex, normal);

    Ray::new(
        Point3::new(x, y, normal.z() * sphere_radius + sphere_center),
        frame.to_world(&temp_direction).normalized(),
    )
}

pub fn camera_space_to_sphere(ray_in: Ray, sphere_center: f32, sphere_radius: f32) -> SphereRay {
    let [x, y, z, _]: [f32; 4] = ray_in.origin.0.into();
    let normal = Vec3::new(x, y, (z - sphere_center).abs()) / sphere_radius;
    let temp_direction = ray_in.direction.normalized();
    let ex = Vec3::new(normal.z(), 0.0, -normal.x());
    let frame = TangentFrame::from_tangent_and_normal(ex, normal);
    SphereRay {
        0: shuffle!(
            ray_in.origin.0,
            frame.to_local(&temp_direction).0,
            [0, 1, 4, 5]
        ),
    }
}

// traces rays from the sensor to the outer pupil
pub fn trace_forward<F>(
    assembly: &LensAssembly,
    zoom: f32,
    input: &Input<Ray>,
    atmosphere_ior: f32,
    aperture_hook: F,
) -> Option<Output<Ray>>
where
    F: Fn(Ray) -> (bool, bool),
{
    assert!(assembly.lenses.len() > 0);
    let mut error = 0;
    let mut n1 = spectrum_eta_from_abbe_num(
        assembly.lenses.last().unwrap().ior,
        assembly.lenses.last().unwrap().vno,
        input.lambda,
    );
    let mut ray = input.ray;
    let mut intensity = 1.0;
    let total_thickness = assembly.total_thickness_at(zoom);
    let mut position = -total_thickness;
    for (k, lens) in assembly.lenses.iter().rev().enumerate() {
        let r = -lens.radius;
        let thickness = lens.thickness_at(zoom);
        position += thickness;
        if lens.lens_type == LensType::Aperture {
            match aperture_hook(ray) {
                (false, true) => {
                    // not blocked by aperture, but still should return early
                    return Some(Output {
                        ray,
                        tau: intensity,
                    });
                }
                (_, _) => {
                    // blocked by aperture (and so no need to trace more) or should return early
                    return None;
                }
            }
        }
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, position + r, lens.housing_radius).unwrap();
        } else if lens.aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                position + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )
            .unwrap();
        } else {
            res = trace_spherical(ray, r, position + r, lens.housing_radius).unwrap();
        }
        ray = res.0;
        let normal = res.1;
        let n2 = if k > 0 {
            spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda)
        } else {
            atmosphere_ior
        };
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return None;
        }
        // not sure why this normalize is here.
        ray.direction = ray.direction.normalized();
        n1 = n2;
    }
    Some(Output {
        ray,
        tau: intensity,
    })
}

// evaluate scene to sensor. input ray must be facing away from the camera.
pub fn trace_reverse<F>(
    assembly: &LensAssembly,
    zoom: f32,
    input: &Input<Ray>,
    atmosphere_ior: f32,
    aperture_hook: F,
) -> Option<Output<Ray>>
where
    F: Fn(Ray) -> (bool, bool),
{
    assert!(assembly.lenses.len() > 0);
    let mut error = 0;
    let mut n1 = atmosphere_ior;
    let mut ray = input.ray;
    let mut intensity = 1.0;
    let mut distsum = 0.0;
    ray.direction = -ray.direction;
    for (_k, lens) in assembly.lenses.iter().enumerate() {
        if lens.lens_type == LensType::Aperture {
            match aperture_hook(ray) {
                (false, true) => {
                    return Some(Output {
                        ray,
                        tau: intensity,
                    });
                }
                (_, _) => {
                    // blocked by aperture and should return
                    return None;
                }
            }
        }
        let r = -lens.radius;

        let dist = lens.thickness_at(zoom);
        let res: (Ray, Vec3);
        if lens.anamorphic {
            res = trace_cylindrical(ray, r, distsum + r, lens.housing_radius).unwrap();
        } else if lens.aspheric > 0 {
            res = trace_aspherical(
                ray,
                r,
                distsum + r,
                lens.aspheric,
                lens.correction,
                lens.housing_radius,
            )
            .unwrap();
        } else {
            res = trace_spherical(ray, r, distsum + r, lens.housing_radius).unwrap();
        }
        ray = res.0;
        let normal = res.1;
        let n2 = spectrum_eta_from_abbe_num(lens.ior, lens.vno, input.lambda);
        // if we were to implement reflection as well, it would probably be here and would probably be probabilistic
        let res = refract(n1, n2, normal, ray.direction);
        ray.direction = res.0;
        intensity *= res.1;
        if intensity < INTENSITY_EPS {
            error |= 8;
        }
        if error > 0 {
            return None;
        }
        distsum -= dist;
        n1 = n2;
    }
    Some(Output {
        ray,
        tau: intensity,
    })
}
