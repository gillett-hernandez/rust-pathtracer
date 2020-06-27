use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;

use std::marker::{Send, Sync};

pub type CameraId = u8;

#[derive(Debug, Copy, Clone)]
pub enum Camera {
    SimpleCamera(SimpleCamera),
}

impl Camera {
    pub fn get_ray(&self, s: f32, t: f32) -> Ray {
        match self {
            Camera::SimpleCamera(inner) => inner.get_ray(s, t),
        }
    }
    pub fn with_aspect_ratio(&self, aspect_ratio: f32) -> Self {
        match self {
            Camera::SimpleCamera(inner) => {
                Camera::SimpleCamera(inner.with_aspect_ratio(aspect_ratio))
            }
        }
    }
    pub fn get_surface(&self) -> Option<Instance> {
        match self {
            Camera::SimpleCamera(inner) => inner.get_surface(),
        }
    }
    pub fn get_pixel_for_ray(&self, ray: Ray) -> Option<(f32, f32)> {
        match self {
            Camera::SimpleCamera(inner) => inner.get_pixel_for_ray(ray),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SimpleCamera {
    pub origin: Point3,
    pub direction: Vec3,
    half_height: f32,
    half_width: f32,
    focal_distance: f32,
    lower_left_corner: Point3,
    vfov: f32,
    pub surface: Instance,
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
    ) -> SimpleCamera {
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

        let transform = Transform3::stack(
            // Some(Transform3::translation(look_from.into())),
            None,
            Some(TangentFrame::new(u, v, w).into()),
            None,
        );

        SimpleCamera {
            origin: look_from,
            direction,
            half_height,
            half_width,
            focal_distance: focus_dist,
            lower_left_corner: look_from
                - u * half_width * focus_dist
                - v * half_height * focus_dist
                - w * focus_dist,
            vfov: vertical_fov,
            surface: Instance::new(
                Aggregate::from(Disk::new(
                    aperture / 2.0,
                    look_from,
                    true,
                    MaterialId::Camera(0),
                    0,
                )),
                Some(transform),
                None,
                None,
            ),
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
    pub fn get_surface(&self) -> Option<Instance> {
        Some(self.surface)
    }
}

impl SimpleCamera {
    pub fn get_ray(&self, s: f32, t: f32) -> Ray {
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
    // returns None if the point on the lens was not from a valid pixel
    pub fn get_pixel_for_ray(&self, ray: Ray) -> Option<(f32, f32)> {
        // would require tracing ray backwards, but for now, try and see what image uv it went through according to the thinlens approximation
        // thinlens says that the ray {self.origin + offset, (self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset}
        // was generated for point (s, t) on the film
        // we know that this ray hit a certain point on the lens. this alone doesn't help determine what pixel coordinate should be returned
        // let self_origin_and_offset = ray.origin;
        let transform = self.surface.transform.expect("somehow camera lens was created without a transform, which should never happen for SimpleCamera");
        let offset_in_uvw_space: Vec3 = transform / (ray.origin - self.origin);
        let (rdx, rdy) = (offset_in_uvw_space.x(), offset_in_uvw_space.y());
        if rdx * rdx + rdy * rdy > self.lens_radius * self.lens_radius {
            None
        } else {
            print!("+");
            // intersect "ray" with image plane
            let local_wi: Vec3 = transform / ray.direction;
            let local_ray_z = local_wi.z();
            let plane_z = self.focal_distance;
            let t = plane_z / local_ray_z;
            if t < 0.0 {
                return None;
            }
            let local_ray = Ray::new(Point3::from(offset_in_uvw_space), local_wi);
            let point_on_image_plane = local_ray.point_at_parameter(t);

            let (s, t) = (
                (point_on_image_plane.x() - self.half_width * self.focal_distance)
                    / (2.0 * self.half_width * self.focal_distance),
                (point_on_image_plane.y() - self.half_height * self.focal_distance)
                    / (2.0 * self.half_height * self.focal_distance),
            );

            if s >= 1.0 || s < 0.0 || t >= 1.0 || t < 0.0 {
                return None;
            }

            Some((s, t))
        }
    }
    pub fn with_aspect_ratio(mut self, aspect_ratio: f32) -> Self {
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
        self
    }
}

unsafe impl Send for Camera {}
unsafe impl Send for SimpleCamera {}
unsafe impl Sync for Camera {}
unsafe impl Sync for SimpleCamera {}

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
