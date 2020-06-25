use crate::math::*;
use std::marker::{Send, Sync};

pub trait Camera: Send + Sync {
    fn get_ray(&self, s: f32, t: f32) -> Ray;
    fn modify_aspect_ratio(&mut self, aspect_ratio: f32);
}

#[derive(Copy, Clone, Debug)]
pub struct SimpleCamera {
    pub origin: Point3,
    pub direction: Vec3,
    focal_distance: f32,
    lower_left_corner: Point3,
    vfov: f32,
    pub horizontal: Vec3,
    pub vertical: Vec3,
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
    ) -> Self {
        let direction = (look_at - look_from).normalized();
        let _lens_radius = aperture / 2.0;
        // vertical_fov should be given in degrees, since it is converted to radians
        let theta: f32 = vertical_fov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        // aspect ratio = half_width / half_height
        let w = -direction;
        let u = -v_up.cross(w).normalized();
        let v = w.cross(u).normalized();

        SimpleCamera {
            origin: look_from,
            direction,
            focal_distance: focus_dist,
            lower_left_corner: look_from
                - u * half_width * focus_dist
                - v * half_height * focus_dist
                - w * focus_dist,
            vfov: vertical_fov,
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
}

unsafe impl Send for SimpleCamera {}
unsafe impl Sync for SimpleCamera {}

impl Camera for SimpleCamera {
    fn get_ray(&self, s: f32, t: f32) -> Ray {
        // circular aperture/lens
        let rd: Vec3 = self.lens_radius * random_in_unit_disk(Sample2D::new_random_sample());
        let offset = self.u * rd.x() + self.v * rd.y();
        let time: f32 = self.t0 + random() * (self.t1 - self.t0);
        let ray_origin: Point3 = self.origin + offset;
        // println!("{:?}", self);
        let ray_direction = (self.lower_left_corner + s * self.horizontal + t * self.vertical
            - ray_origin)
            .normalized();
        assert!(ray_origin.is_normal());
        assert!(ray_direction.is_normal());
        Ray::new_with_time(ray_origin, ray_direction, time)
    }
    fn modify_aspect_ratio(&mut self, aspect_ratio: f32) {
        assert!(self.focal_distance > 0.0 && self.vfov > 0.0);
        let theta: f32 = self.vfov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        self.lower_left_corner = self.origin
            - self.u * half_width * self.focal_distance
            - self.v * half_height * self.focal_distance
            - self.w * self.focal_distance;
        self.horizontal = self.u * 2.0 * half_width * self.focal_distance;
        self.vertical = self.v * 2.0 * half_height * self.focal_distance;
    }
}

// pub enum BundledCamera {
//     SimpleCamera(SimpleCamera),
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera() {
        let camera: SimpleCamera = SimpleCamera::new(
            Point3::new(-100.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            45.0,
            0.6,
            1.0,
            1.0,
            0.0,
            1.0,
        );
        println!("camera origin {:?}", camera.origin);
        println!("camera direction {:?}", camera.direction);
        let s = random();
        let t = random();
        let r: Ray = camera.get_ray(s, t);
        println!("ray origin {:?}", r.origin);
        println!("ray direction {:?}", r.direction);
        assert!(
            r.direction * Vec3::X > 0.0,
            "x component of direction of camera ray pointed wrong"
        );
    }
}
