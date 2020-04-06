use crate::math::{Vec3, PI};

pub fn random() -> f32 {
    rand::random()
}
pub fn random_in_unit_sphere() -> Vec3 {
    let u = random() * PI * 2.0;
    let v = (2.0 * random() - 1.0).acos();
    let w = random().powf(1.0 / 3.0);
    Vec3::new(u.cos() * v.sin() * w, v.cos() * w, u.sin() * v.sin() * w)
}

pub fn random_in_unit_disk() -> Vec3 {
    let u: f32 = random() * PI * 2.0;
    let v: f32 = random().powf(1.0 / 2.0);
    Vec3::new(u.cos() * v, u.sin() * v, 0.0)
}

pub fn random_cosine_direction() -> Vec3 {
    let r1: f32 = random();
    let r2: f32 = random();
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
