use crate::math::*;
pub struct Camera {
    origin: Point3,
    direction: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
    t0: f32,
    t1: f32,
}
impl Camera {
    pub fn get_ray(s: f32, t: f32) -> Ray {
        // circular aperture
        let rd: Vec3 = lens_radius * random_in_unit_disk();
        let offset = u * rd.x() + v * rd.y();
        let time: f32 = time0 + random_double() * (time1 - time0);
        Ray::new(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            time,
        )
    }
}
