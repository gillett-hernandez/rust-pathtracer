use crate::geometry::*;
use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct ProjectiveCamera {
    pub origin: Point3,
    pub direction: Vec3,
    // half_height: f32,
    // half_width: f32,
    focal_distance: f32,
    lower_left_corner: Point3,
    vfov: f32,
    pub lens: Instance,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    // TODO: change this to aperture from rust_optics crate
    aperture_radius: f32,
}

impl ProjectiveCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        vertical_fov: f32, // vertical_fov should be given in degrees, since it is converted to radians
        focal_distance: f32,
        aperture: f32,
    ) -> ProjectiveCamera {
        let direction = (look_at - look_from).normalized();
        let lens_radius = aperture / 2.0;
        let theta: f32 = vertical_fov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = 1.0 * half_height;

        let aspect_ratio = half_width / half_height;
        info!(
            "aspect ratio on instantiation of ProjectiveCamera = {}",
            aspect_ratio
        );

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
        )
        .inverse();

        if lens_radius == 0.0 {
            warn!("Warn: lens radius is 0.0");
        }

        ProjectiveCamera {
            origin: look_from,
            direction,
            // half_height,
            // half_width,
            focal_distance,
            lower_left_corner: look_from
                - u * half_width * focal_distance
                - v * half_height * focal_distance
                - w * focal_distance,
            vfov: vertical_fov,
            lens: Instance::new(
                Aggregate::from(Disk::new(lens_radius, Point3::ORIGIN, true)),
                Some(transform),
                Some(MaterialId::Camera(0)),
                0,
            ),
            horizontal: u * 2.0 * half_width * focal_distance,
            vertical: v * 2.0 * half_height * focal_distance,
            u,
            v,
            w,
            aperture_radius: aperture / 2.0,
        }
    }
    pub fn get_surface(&self) -> Option<&Instance> {
        Some(&self.lens)
    }
}

impl Camera<f32, f32> for ProjectiveCamera {
    fn get_ray(&self, sampler: &mut Box<dyn Sampler>, _lambda: f32, u: f32, v: f32) -> (Ray, f32) {
        // circular aperture/lens
        let rd: Vec3 = self.aperture_radius * random_in_unit_disk(sampler.draw_2d());
        let offset = self.u * rd.x() + self.v * rd.y();
        let ray_origin: Point3 = self.origin + offset;

        let point_on_plane = self.lower_left_corner + u * self.horizontal + v * self.vertical;

        // println!("point on focal plane {:?}", point_on_plane);
        let ray_direction = (point_on_plane - ray_origin).normalized();
        debug_assert!(ray_origin.is_finite());
        debug_assert!(ray_direction.is_finite());
        (Ray::new_with_time(ray_origin, ray_direction, 0.0), 1.0)
    }
    fn with_aspect_ratio(mut self, aspect_ratio: f32) -> Self {
        assert!(self.focal_distance > 0.0 && self.vfov > 0.0);
        let theta: f32 = self.vfov.to_radians();
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
    // returns None if the point on the lens was not from a valid pixel
    fn get_pixel_for_ray(&self, ray: Ray, _lambda: f32) -> Option<(f32, f32)> {
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

        let (plane_width, plane_height) = (self.horizontal.norm(), self.vertical.norm());

        let (u, v) = (
            point_on_focal_plane.x() / plane_width + 0.5,
            point_on_focal_plane.y() / plane_height + 0.5,
        );

        #[cfg(test)]
        {
            // for helping when debugging.
            // FIXME? remove entire block?
            println!(
                "{:?} {} {} {}",
                point_on_focal_plane,
                plane_width / 2.0,
                plane_height / 2.0,
                plane_width / plane_height
            );
            println!("{} {}", u, v);
        }

        let zero_to_one = 0.0..1.0;

        if !zero_to_one.contains(&u) || !zero_to_one.contains(&v) {
            None
        } else {
            Some((u, v))
        }
    }

    fn eval_we(
        &self,
        _lambda: f32,
        _normal: Vec3,
        _from: Point3,
        _to: Point3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        // TODO
        todo!()
    }

    fn sample_we(
        &self,
        film_sample: Sample2D,
        sampler: &mut Box<dyn Sampler>,
        lambda: f32,
    ) -> (Ray, Vec3, PDF<f32, SolidAngle>) {
        let (ray, tau) = self.get_ray(sampler, lambda, film_sample.x, film_sample.y);
        (ray, self.direction, tau.into())
    }
}

unsafe impl Send for ProjectiveCamera {}
unsafe impl Sync for ProjectiveCamera {}

#[cfg(test)]
mod tests {
    use super::*;
    use math::prelude::*;

    #[test]
    fn test_camera() {
        let camera: ProjectiveCamera = ProjectiveCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            35.2,
            5.0,
            0.08,
        )
        .with_aspect_ratio(0.6);
        let s = debug_random();
        let t = debug_random();
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let (r, _tau) = camera.get_ray(&mut sampler, 550.0, s, t);
        println!("camera ray {:?}", r);
        println!(
            "camera ray in camera local space {:?}",
            camera.lens.transform.unwrap().to_local(r)
        );
        let pixel_uv = camera.get_pixel_for_ray(r, 550.0);
        println!("s and t are actually {} and {}", s, t);
        println!("{:?}", pixel_uv);
    }

    #[test]
    fn test_camera_wide_aspect() {
        let (width, height) = (1920.0, 1080.0);
        let camera: ProjectiveCamera = ProjectiveCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            35.2,
            5.0,
            0.08,
        )
        .with_aspect_ratio(width as f32 / height as f32);
        let px = (0.99 * width) as usize;
        let py = (0.99 * height) as usize;
        let s = (px as f32) / width + debug_random() / width;
        let t = (py as f32) / height + debug_random() / height;
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let (r, _tau) = camera.get_ray(&mut sampler, 550.0, s, t);
        println!("camera ray {:?}", r);
        println!(
            "camera ray in camera local space {:?}",
            camera.lens.transform.unwrap().to_local(r)
        );
        println!("s and t are actually {} and {}", s, t);
        println!("px and py are actually {} and {}", px, py);
        let maybe_pixel_uv = camera.get_pixel_for_ray(r, 550.0);
        println!("calculated pixel uv is {:?}", maybe_pixel_uv);
        if let Some(pixel_uv) = maybe_pixel_uv {
            let px_c = pixel_uv.0 * width;
            let py_c = height - pixel_uv.1 * height;
            println!("calculated pixel is ({}, {})", px_c, py_c);
        }
    }

    #[test]
    fn check_camera_position_and_orientation() {
        use crate::hittable::Hittable;
        let camera: ProjectiveCamera = ProjectiveCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            27.0,
            5.0,
            0.08,
        )
        .with_aspect_ratio(0.6);

        let sample_from = Point3::ORIGIN;

        let camera_surface = camera.get_surface().unwrap();
        let transform = camera_surface.transform.unwrap();
        println!("transform * = {:?}", transform.to_local(sample_from));
        println!("transform / = {:?}", transform.to_world(sample_from));
        let sample = Sample2D::new_random_sample();
        let result = camera_surface.sample(sample, sample_from);
        println!("{:?}", result);
        let to = transform.to_world(Point3::ORIGIN);
        let direction = (to - sample_from).normalized();
        let result2 = camera_surface.psa_pdf(
            (Vec3::X * direction).abs(),
            (Vec3::Z * direction).abs(),
            sample_from,
            to,
        );
        println!("{:?}", result2);
    }
}
