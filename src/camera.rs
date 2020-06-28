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
        let lens_radius = aperture / 2.0;
        // vertical_fov should be given in degrees, since it is converted to radians
        let theta: f32 = vertical_fov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        // aspect ratio = half_width / half_height
        let w = -direction;
        let u = -v_up.cross(w).normalized();
        let v = w.cross(u).normalized();
        println!(
            "constructing camera with point, direction, and uvw = {:?} {:?} {:?} {:?} {:?}",
            look_from, direction, u, v, w
        );

        let transform = Transform3::stack(
            // Some(Transform3::translation(look_from.into())),
            None,
            Some(TangentFrame::new(u, v, -w).into()),
            None,
        );

        if lens_radius == 0.0 {
            println!("Warn: lens radius is 0");
        }

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
                    lens_radius,
                    Point3::ORIGIN,
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
        let point_on_surface = ray.origin;
        let transform = self.surface.transform.expect("somehow camera lens was created without a transform, which should never happen for SimpleCamera");
        let point_in_local_space = Point3::from(point_on_surface - self.origin);
        let ray = Ray::new(point_in_local_space, ray.direction);
        let ray = transform * ray;
        let offset_in_uvw_space: Point3 = ray.origin;
        let (rdx, rdy) = (offset_in_uvw_space.x(), offset_in_uvw_space.y());
        if rdx * rdx + rdy * rdy > self.lens_radius * self.lens_radius {
            println!("rdx, rdy was {}, {}, dist_from_origin = {}, which is larger than self.lens_radius {}", rdx, rdy, (rdx * rdx + rdy * rdy).sqrt(), self.lens_radius);
            None
        } else {
            // intersect "ray" with image plane
            let local_wi: Vec3 = ray.direction;
            let local_ray = Ray::new(offset_in_uvw_space, local_wi);

            let local_ray_z = local_wi.z();
            let plane_z = self.focal_distance;
            let t = plane_z / local_ray_z;
            /*look_from
            - u * half_width * focus_dist
            - v * half_height * focus_dist
            - w * focus_dist*/
            println!("lower left corner {:?}", self.lower_left_corner);
            println!(
                "lower left corner transformed {:?}",
                transform * self.lower_left_corner
            );
            let lower_left_corner = Vec3::new(
                self.half_width * self.focal_distance,
                self.half_height * self.focal_distance,
                self.focal_distance,
            );
            println!("lower left corner vec {:?}", lower_left_corner);
            let point_on_image_plane = local_ray.point_at_parameter(t) - offset_in_uvw_space;
            // let inverse_point_on_image_plane =
            //     local_ray.point_at_parameter(-t) - offset_in_uvw_space;
            if t < 0.0 {
                println!("t was {}", t);
                return None;
            }

            println!("local ray {:?} ", local_ray);
            println!("point on image plane {:?}", point_on_image_plane);
            // println!(
            //     "inverse point on image plane {:?}",
            //     inverse_point_on_image_plane
            // );

            let (s, t) = (-point_on_image_plane.x(), -point_on_image_plane.y());

            if s >= 1.0 || s < -1.0 || t >= 1.0 || t < -1.0 {
                println!("s and t were calculated but were {} and {}", s, t);
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
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            27.0,
            0.6,
            5.0,
            0.08,
            0.0,
            1.0,
        );
        let s = random();
        let t = random();
        let r: Ray = camera.get_ray(s, t);
        println!("ray {:?}", r);
        let pixel_uv = camera.get_pixel_for_ray(r);
        println!("s and t are actually {} and {}", s, t);
        println!("{:?}", pixel_uv);
        if let Some((u, v)) = pixel_uv {
            println!("remapped {} {}", (u + 1.0) / 2.0, (v + 1.0) / 2.0);
        }
    }
    #[test]
    fn test_debug() {
        let look_from = Point3::new(-5.0, 0.0, 0.0);
        let look_at = Point3::ZERO;

        let origin = look_from;
        let lens_radius = 0.01;
        let u = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.0, 0.0, -1.0);
        let w = Vec3::new(-1.0, 0.0, 0.0);
        let focus_dist = 5.0;
        let half_width = 0.5;
        let half_height = 0.3;
        let rd: Vec3 = lens_radius * random_in_unit_disk(Sample2D::new_random_sample());
        let offset = u * rd.x() + v * rd.y();
        let ray_origin: Point3 = origin + offset;
        // println!("{:?}", ;
        let s = 0.3;
        let t = 0.7;
        let u_halfwidth_focus_dist = u * half_width * focus_dist;
        let v_halfheight_focus_dist = v * half_height * focus_dist;
        let ray_direction = (u_halfwidth_focus_dist * (s * 2.0 - 1.0)
            + v_halfheight_focus_dist * (t * 2.0 - 1.0)
            - w * focus_dist
            - offset)
            .normalized();
        // let un_normalized = Vec3::new()
    }
}
