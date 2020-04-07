use crate::math::{Sample2D, Sample3D, Vec3, PI};

pub fn random() -> f32 {
    rand::random()
}

pub fn random_in_unit_sphere(r: Sample3D) -> Vec3 {
    let u = r.x * PI * 2.0;
    let v = (2.0 * r.y - 1.0).acos();
    let w = r.z.powf(1.0 / 3.0);
    Vec3::new(u.cos() * v.sin() * w, v.cos() * w, u.sin() * v.sin() * w)
}

pub fn random_in_unit_disk(r: Sample2D) -> Vec3 {
    let u: f32 = r.x * PI * 2.0;
    let v: f32 = r.y.powf(1.0 / 2.0);
    Vec3::new(u.cos() * v, u.sin() * v, 0.0)
}

pub fn random_cosine_direction(r: Sample2D) -> Vec3 {
    let r1: f32 = r.x;
    let r2: f32 = r.y;
    let z: f32 = (1.0 - r2).sqrt();
    let phi: f32 = 2.0 * PI * r1;
    let x: f32 = phi.cos() * r2.sqrt();
    let y: f32 = phi.sin() * r2.sqrt();
    Vec3::new(x, y, z)
}

// pub fn random_to_sphere(radius: f32, distance_squared: f32) -> Vec3 {
//     let r1: f32 = random();
//     let r2: f32 = random();
//     let z: f32 = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);
//     let phi: f32 = 2 * M_PI * r1;
//     let x: f32 = cos(phi) * sqrt(1 - z * z);
//     let y: f32 = sin(phi) * sqrt(1 - z * z);
//     return vec3(x, y, z);
// }
