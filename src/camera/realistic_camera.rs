use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;

use optics::{lens_sampler::RadialSampler, LensAssembly, LensInterface};

#[derive(Debug, Clone)]
pub struct RealisticCamera {
    pub origin: Point3,
    pub direction: Vec3,
    focal_distance: f32,
    vfov: f32,
    pub lens: Instance,
    pub assembly: LensAssembly,
    transform: Transform3,
    lens_radius: f32,
    t0: f32,
    t1: f32,
}

impl RealisticCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        focal_distance: f32,
        aperture: f32,
        interfaces: Vec<LensInterface>,
        t0: f32,
        t1: f32,
    ) -> RealisticCamera {
        let direction = (look_at - look_from).normalized();
        let lens_radius = aperture / 2.0;
        // vertical_fov should be given in degrees, since it is converted to radians
        let theta: f32 = vertical_fov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        #[cfg(test)]
        {
            let aspect_ratio = half_width / half_height;
            println!("{}", aspect_ratio);
        }
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
            println!("Warn: lens radius is 0");
        }

        RealisticCamera {
            origin: look_from,
            direction,
            focal_distance,
            vfov: vertical_fov,
            lens: Instance::new(
                Aggregate::from(Disk::new(lens_radius, Point3::ORIGIN, true)),
                Some(transform),
                Some(MaterialId::Camera(0)),
                0,
            ),
            assembly: LensAssembly::new(&interfaces),
            transform,
            lens_radius: aperture / 2.0,
            t0,
            t1,
        }
    }
    pub fn get_surface(&self) -> Option<&Instance> {
        Some(&self.lens)
    }
}

impl RealisticCamera {
    pub fn get_ray(&self, sample: Sample2D, s: f32, t: f32) -> Ray {
        // circular aperture/lens
        let time: f32 = self.t0 + random() * (self.t1 - self.t0);
        let ray_origin: Point3 = self.origin;
        Ray::new(ray_origin, Vec3::Z)
    }
    // returns None if the point on the lens was not from a valid pixel
    // pub fn get_pixel_for_ray(&self, ray: Ray) -> Option<(f32, f32)> {
    //     // would require tracing ray backwards, but for now, try and see what image uv it went through according to the thinlens approximation

    //     // println!("ray is {:?}", ray);
    //     let ray_in_local_space = self.lens.transform.unwrap().to_local(ray);
    //     // let ray_in_local_space = self.lens.transform.unwrap() * ray;
    //     // println!("ray in local space is {:?}", ray_in_local_space);

    //     // trace ray in local space to intersect with virtual focal plane

    //     let plane_z = self.focal_distance;

    //     let t = -plane_z / ray_in_local_space.direction.z();
    //     // let t = 0.0;

    //     let point_on_focal_plane = ray_in_local_space.point_at_parameter(t);

    //     let (plane_width, plane_height) = (self.horizontal.norm(), self.vertical.norm());

    //     #[cfg(test)]
    //     println!(
    //         "{:?} {} {} {}",
    //         point_on_focal_plane,
    //         plane_width / 2.0,
    //         plane_height / 2.0,
    //         plane_width / plane_height
    //     );

    //     let (u, v) = (
    //         point_on_focal_plane.x() / plane_width + 0.5,
    //         point_on_focal_plane.y() / plane_height + 0.5,
    //     );

    //     #[cfg(test)]
    //     println!("{} {}", u, v);

    //     if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 {
    //         None
    //     } else {
    //         Some((u, v))
    //     }
    // }
    // pub fn with_aspect_ratio(mut self, aspect_ratio: f32) -> Self {
    //     assert!(self.focal_distance > 0.0 && self.vfov > 0.0);
    //     let theta: f32 = self.vfov.to_radians();
    //     let half_height = (theta / 2.0).tan();
    //     let half_width = aspect_ratio * half_height;
    //     self
    // }
}

unsafe impl Send for RealisticCamera {}
unsafe impl Sync for RealisticCamera {}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_camera_wide_aspect() {
    //     let (width, height) = (1920.0, 1080.0);
    //     let camera: RealisticCamera = RealisticCamera::new(
    //         Point3::new(-5.0, 0.0, 0.0),
    //         Point3::ZERO,
    //         Vec3::Z,
    //         35.2,
    //         width as f32 / height as f32,
    //         5.0,
    //         0.08,
    //         0.0,
    //         1.0,
    //     );
    //     let px = (0.99 * width) as usize;
    //     let py = (0.99 * height) as usize;
    //     let s = (px as f32) / width + random() / width;
    //     let t = (py as f32) / height + random() / height;
    //     let r: Ray = camera.get_ray(
    //         Sample2D {
    //             x: random(),
    //             y: random(),
    //         },
    //         s,
    //         t,
    //     );
    //     println!("camera ray {:?}", r);
    //     println!(
    //         "camera ray in camera local space {:?}",
    //         camera.lens.transform.unwrap().to_local(r)
    //     );
    //     println!("s and t are actually {} and {}", s, t);
    //     println!("px and py are actually {} and {}", px, py);
    //     let maybe_pixel_uv = camera.get_pixel_for_ray(r);
    //     println!("calculated pixel uv is {:?}", maybe_pixel_uv);
    //     if let Some(pixel_uv) = maybe_pixel_uv {
    //         let px_c = pixel_uv.0 * width;
    //         let py_c = height - pixel_uv.1 * height;
    //         println!("calculated pixel is ({}, {})", px_c, py_c);
    //     }
    // }

    #[test]
    fn check_camera_position_and_orientation() {
        use crate::hittable::Hittable;
        let camera: RealisticCamera = RealisticCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            27.0,
            0.6,
            5.0,
            0.08,
            vec![],
            0.0,
            1.0,
        );

        let sample_from = Point3::ORIGIN;

        let camera_surface = camera.get_surface().unwrap();
        let transform = camera_surface.transform.unwrap();
        println!("transform * = {:?}", transform.to_local(sample_from));
        println!("transform / ={:?}", transform.to_world(sample_from));
        let sample = Sample2D::new_random_sample();
        let result = camera_surface.sample(sample, sample_from);
        println!("{:?}", result);
        let to = transform.to_world(Point3::ORIGIN);
        let result2 =
            camera_surface.psa_pdf(Vec3::X * (to - sample_from).normalized(), sample_from, to);
        println!("{:?}", result2);
    }
}
