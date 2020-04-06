use crate::math::*;
pub struct SimpleCamera {
    origin: Point3,
    direction: Vec3,
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

impl SimpleCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        focus_dist: f32,
        aperture: f32,
        t0: f32,
        t1: f32,
    ) -> SimpleCamera {
        let direction = look_at - look_from;
        let lens_radius = aperture / 2.0;
        // vertical_fov should be given in degrees, since it is converted to radians
        let theta: f32 = vertical_fov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        let w = direction.normalized();
        let u = v_up.cross(w).normalized();
        let v = w.cross(u);

        SimpleCamera {
            origin: look_from,
            direction,
            lower_left_corner: look_from
                - u * half_width * focus_dist
                - v * half_height * focus_dist
                - w * focus_dist,
            horizontal: u * 2.0 * half_width * focus_dist,
            vertical: v * 2.0 * half_height * focus_dist,
            u,
            v,
            w,
            lens_radius: aperture / 2.0,
            t0,
            t1,
        }
    }
    pub fn get_ray(&self, s: f32, t: f32) -> Ray {
        // circular aperture/lens
        let rd: Vec3 = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        let time: f32 = self.t0 + random() * (self.t1 - self.t0);
        Ray::new_with_time(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
            time,
        )
    }
}
