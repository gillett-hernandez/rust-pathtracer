use std::f32::consts::SQRT_2;

use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;

use optics::{
    bladed_aperture, lens_sampler::RadialSampler, Input, LensAssembly, LensInterface, Output,
};
use spectral::EXTENDED_VISIBLE_RANGE;

#[derive(Debug, Clone)]
pub struct RealisticCamera {
    pub origin: Point3,
    pub direction: Vec3,
    focal_adjustment: f32,
    pub lens: Instance,
    pub assembly: LensAssembly,
    sampler: RadialSampler,
    aspect_ratio: f32, // 16:9 would yield an aspect ratio of 16/9
    sensor_size: f32,
    film_position: f32,
    aperture_radius: f32,
    transform: Transform3,
    lens_zoom: f32,
    lens_radius: f32,
    t0: f32,
    t1: f32,
}

impl RealisticCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        focal_adjustment: f32,
        sensor_size: f32,
        f_stop: f32,
        lens_zoom: f32,
        interfaces: Vec<LensInterface>,
        t0: f32,
        t1: f32,
        radial_bins: usize,
        wavelength_bins: usize,
        solver_heat: f32,
    ) -> RealisticCamera {
        println!("{}", "constructing realistic camera");
        let direction = (look_at - look_from).normalized();
        // vertical_fov should be given in degrees, since it is converted to radians
        let mut assembly = LensAssembly::new(&interfaces);
        let mut aperture_radius = assembly.lenses[assembly.aperture_index].housing_radius;
        aperture_radius /= f_stop;
        assembly.lenses[assembly.aperture_index].housing_radius = aperture_radius;
        let lens_radius = assembly.lenses[0].housing_radius;

        let w = -direction;
        let u = -v_up.cross(w).normalized();
        let v = w.cross(u).normalized();
        let film_position = -assembly.total_thickness_at(lens_zoom) + focal_adjustment;

        // scale down, rotate, and move
        let transform = Transform3::from_stack(
            Some(Transform3::from_scale(Vec3::new(1000.0, 1000.0, 1000.0))),
            Some(TangentFrame::new(u, -v, -w).into()), // rotate and stuff
            Some(Transform3::from_translation(Point3::ORIGIN - look_from)), // move to match camera origin
        )
        .inverse();
        println!("{}", "creating cache");
        let sampler = RadialSampler::new(
            SQRT_2 * sensor_size / 2.0, // diagonal.
            radial_bins,
            wavelength_bins,
            EXTENDED_VISIBLE_RANGE,
            film_position,
            &assembly,
            lens_zoom,
            |aperture_radius, ray| bladed_aperture(aperture_radius, 6, ray),
            solver_heat,
            sensor_size,
        );
        println!("{}", "finished cache");

        RealisticCamera {
            origin: look_from,
            direction,
            focal_adjustment,

            lens: Instance::new(
                Aggregate::from(Disk::new(lens_radius, Point3::ORIGIN, true)),
                Some(transform),
                Some(MaterialId::Camera(0)),
                0,
            ),
            aspect_ratio: 1.0,
            sensor_size,
            film_position,
            lens_zoom,
            assembly,
            aperture_radius,
            sampler,
            transform,
            lens_radius,
            t0,
            t1,
        }
    }
    pub fn get_surface(&self) -> Option<&Instance> {
        Some(&self.lens)
    }
}

impl RealisticCamera {
    pub fn get_ray(
        &self,
        sampler: &mut Box<dyn Sampler>,
        lambda: f32,
        s: f32,
        t: f32,
    ) -> (Ray, f32) {
        // circular aperture/lens
        let _time: f32 = self.t0 + random() * (self.t1 - self.t0);
        // crop sensor to match aspect ratio
        let (x_factor, y_factor) = if self.aspect_ratio > 1.0 {
            // x larger than y
            // thus x needs to be scaled down
            (1.0, self.aspect_ratio)
        } else {
            // y larger than x
            // thus x is scaled down
            (1.0 / self.aspect_ratio, 1.0)
        };
        let central_point: Point3 = Point3::new(
            (s - 0.5) * self.sensor_size * x_factor,
            (t - 0.5) * self.sensor_size * y_factor,
            self.film_position,
        );
        let mut _attempts = 0;
        let mut result = None;

        for _ in 0..100 {
            // let s0 = sampler.draw_2d();
            // let [x, y, z, _]: [f32; 4] = central_point.0.into();
            // x += (s0.x - 0.5) / width as f32 * self.sensor_size;
            // y += (s0.y - 0.5) / height as f32 * self.sensor_size;

            // let point = Point3::new(x, y, z);
            let point = central_point;
            let v = self
                .sampler
                .sample(lambda, point, sampler.draw_2d(), sampler.draw_1d());
            let ray = Ray::new(point, v);

            _attempts += 1;
            // do actual tracing through lens for film sample
            let trace_result =
                self.assembly
                    .trace_forward(self.lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                        (bladed_aperture(self.aperture_radius, 6, e), false)
                    });
            if let Some(Output {
                ray: mut pupil_ray,
                tau,
            }) = trace_result
            {
                // scale back down to meters
                // pupil_ray.origin = Point3::from(Vec3::from(pupil_ray.origin) / 1000.0);
                pupil_ray = self.transform.to_world(pupil_ray);
                pupil_ray.direction = pupil_ray.direction.normalized();
                result = Some((pupil_ray, tau));
                break;
            }
        }
        if let Some(r) = result {
            r
        } else {
            // println!(
            //     "{:?}, {}, {:?}",
            //     central_point,
            //     lambda,
            //     self.sampler
            //         .sample(lambda, central_point, sampler.draw_2d(), sampler.draw_1d())
            // );

            // panic!();
            (Ray::new(central_point, Vec3::Z), 0.0)
        }
    }
    // returns None if the point on the lens was not from a valid pixel
    pub fn get_pixel_for_ray(&self, _ray: Ray, _lambda: f32) -> Option<(f32, f32)> {
        // TODO: implement backwards tracing. requires a reversed radial sampler cache or something like an OMP fit.
        todo!();
    }
    pub fn with_aspect_ratio(mut self, aspect_ratio: f32) -> Self {
        self.aspect_ratio = aspect_ratio;
        self
    }
}

unsafe impl Send for RealisticCamera {}
unsafe impl Sync for RealisticCamera {}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read};

    use optics::parse_lenses_from;

    use super::*;

    fn read_from_file(path: &str) -> Vec<LensInterface> {
        let mut camera_file = File::open(path).unwrap();
        let mut camera_spec = String::new();
        camera_file.read_to_string(&mut camera_spec).unwrap();
        let (interfaces, _n0, _n1) = parse_lenses_from(&camera_spec);
        interfaces
    }

    #[test]
    fn test_generate_ray() {
        let interfaces = read_from_file("data/cameras/petzval_kodak.txt");
        assert!(interfaces.len() > 0);
        let camera: RealisticCamera = RealisticCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            10.0,
            35.0,
            6.0,
            0.0,
            interfaces,
            0.0,
            1.0,
            128,
            128,
            0.01,
        );
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let r = camera.get_ray(&mut sampler, 550.0, 0.5, 0.5);
        println!("{:?}", r);
    }

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
        let interfaces = read_from_file("data/cameras/petzval_kodak.fx");
        let camera: RealisticCamera = RealisticCamera::new(
            Point3::new(-5.0, 0.0, 0.0),
            Point3::ZERO,
            Vec3::Z,
            -10.0,
            35.0,
            10.0,
            0.0,
            interfaces,
            0.0,
            1.0,
            128,
            128,
            0.01,
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
