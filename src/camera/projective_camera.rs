use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;
#[derive(Debug, Copy, Clone)]
pub struct ProjectiveCamera {
    pub origin: Point3,
    pub direction: Vec3,
    half_height: f32,
    half_width: f32,
    focal_distance: f32,
    lower_left_corner: Point3,
    vfov: f32,
    pub lens: Instance,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
    t0: f32,
    t1: f32,
}

impl ProjectiveCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        focal_distance: f32,
        aperture: f32,
        t0: f32,
        t1: f32,
    ) -> ProjectiveCamera {
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
        // println!(
        //     "constructing camera with point, direction, and uvw = {:?} {:?} {:?} {:?} {:?}",
        //     look_from, direction, u, v, w
        // );

        let transform = Transform3::from_stack(
            None,
            Some(TangentFrame::new(u, -v, w).into()), // rotate and stuff
            Some(Transform3::from_translation(Point3::ORIGIN - look_from)), // move to match camera origin
        );

        if lens_radius == 0.0 {
            println!("Warn: lens radius is 0");
        }

        ProjectiveCamera {
            origin: look_from,
            direction,
            half_height,
            half_width,
            focal_distance,
            lower_left_corner: look_from
                - u * half_width * focal_distance
                - v * half_height * focal_distance
                - w * focal_distance,
            vfov: vertical_fov,
            lens: Instance::new(
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
            horizontal: u * 2.0 * half_width * focal_distance,
            vertical: v * 2.0 * half_height * focal_distance,
            u,
            v,
            w,
            lens_radius: aperture / 2.0,
            t0,
            t1,
        }
    }
    pub fn get_surface(&self) -> Option<Instance> {
        Some(self.lens)
    }
}

impl ProjectiveCamera {
    pub fn get_ray(&self, sample: Sample2D, s: f32, t: f32) -> Ray {
        // circular aperture/lens
        let rd: Vec3 = self.lens_radius * random_in_unit_disk(sample);
        let offset = self.u * rd.x() + self.v * rd.y();
        let time: f32 = self.t0 + random() * (self.t1 - self.t0);
        let ray_origin: Point3 = self.origin + offset;

        let point_on_plane = self.lower_left_corner + s * self.horizontal + t * self.vertical;

        // println!("point on focal plane {:?}", point_on_plane);
        let ray_direction = (point_on_plane - ray_origin).normalized();
        debug_assert!(ray_origin.is_normal());
        debug_assert!(ray_direction.is_normal());
        Ray::new_with_time(ray_origin, ray_direction, time)
    }
    // returns None if the point on the lens was not from a valid pixel
    pub fn get_pixel_for_ray(&self, ray: Ray) -> Option<(f32, f32)> {
        // would require tracing ray backwards, but for now, try and see what image uv it went through according to the thinlens approximation

        // println!("ray is {:?}", ray);
        let ray_in_local_space = self.lens.transform.unwrap().to_local(ray);
        // let ray_in_local_space = self.lens.transform.unwrap() * ray;
        // println!("ray in local space is {:?}", ray_in_local_space);

        // trace ray in local space to intersect with virtual focal plane

        let plane_z = self.focal_distance;

        let t = -plane_z / ray_in_local_space.direction.z();
        // let t = 0.0;

        let point_on_focal_plane = ray_in_local_space.point_at_parameter(t);

        let (plane_width, plane_height) = (
            self.focal_distance * self.half_width * 2.0,
            self.focal_distance * self.half_height * 2.0,
        );

        let (u, v) = (
            (point_on_focal_plane.x() + plane_width / 2.0) / plane_width,
            (point_on_focal_plane.y() + plane_height / 2.0) / plane_height,
        );

        if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 {
            None
        } else {
            Some((u, v))
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

unsafe impl Send for ProjectiveCamera {}
unsafe impl Sync for ProjectiveCamera {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera() {
        let camera: ProjectiveCamera = ProjectiveCamera::new(
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
        let r: Ray = camera.get_ray(
            Sample2D {
                x: random(),
                y: random(),
            },
            s,
            t,
        );
        println!("camera ray {:?}", r);
        println!(
            "camera ray in camera local space {:?}",
            camera.lens.transform.unwrap() * r
        );
        let pixel_uv = camera.get_pixel_for_ray(r);
        println!("s and t are actually {} and {}", s, t);
        println!("{:?}", pixel_uv);
    }

    #[test]
    fn check_camera_position_and_orientation() {
        use crate::hittable::Hittable;
        let camera: ProjectiveCamera = ProjectiveCamera::new(
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

        let sample_from = Point3::ORIGIN;

        let camera_surface = camera.get_surface().unwrap();
        let transform = camera_surface.transform.unwrap();
        println!("transform * = {:?}", transform * sample_from);
        println!("transform / ={:?}", transform / sample_from);
        let sample = Sample2D::new_random_sample();
        let result = camera_surface.sample(sample, sample_from);
        println!("{:?}", result);
        let result2 = camera_surface.pdf(Vec3::X, sample_from, transform.to_world(Point3::ORIGIN));
        println!("{:?}", result2);
    }
}
