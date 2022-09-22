use crate::prelude::*;

use crate::geometry::*;

#[derive(Debug, Clone)]
pub struct PanoramaCamera {
    pub origin: Point3,
    pub direction: Vec3, // primary facing direction in world space
    // angle_span.0 is horizontal, and must be below 2pi radians
    // angle_span.1 is vertical, and must be below pi radians
    pub angle_span: (f32, f32),
    // TODO: figure out whether it's possible to have an aperture for a panoramic camera.
    // might need one in order for a panoramic camera to work in BDPT/LT, otherwise the probability of sampling or intersecting the camera from the scene would be 0 everywhere, or at least a delta distribution.
    surface: Instance,
    transform: Transform3,
}

impl PanoramaCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        horizontal_fov: f32, // horizontal_fov should be given in degrees, since it is converted to radians
        vertical_fov: f32,   // ditto for this
    ) -> PanoramaCamera {
        let direction = (look_at - look_from).normalized();

        let w = direction;
        let u = v_up.cross(w).normalized();
        let v = w.cross(u).normalized();

        let (horizontal_fov, vertical_fov) = (
            horizontal_fov.to_radians().clamp(0.0, TAU),
            vertical_fov.to_radians().clamp(0.0, PI),
        );

        info!(
            "constructing camera with point, direction, and uvw = {:?} {:?} {:?} {:?} {:?}",
            look_from, direction, u, v, w
        );

        let transform = Transform3::from_stack(
            None,
            Some(TangentFrame::new(u, v, w).into()), // rotate and stuff
            Some(Transform3::from_translation(Point3::ORIGIN - look_from)), // move to match camera origin
        )
        .inverse();

        PanoramaCamera {
            origin: look_from,
            direction,
            angle_span: (horizontal_fov, vertical_fov),
            surface: Instance::new(
                Aggregate::from(Sphere::new(1.0, Point3::ORIGIN)),
                Some(transform),
                Some(MaterialId::Camera(0)),
                0,
            ),
            transform,
        }
    }
    pub fn get_surface(&self) -> Option<&Instance> {
        Some(&self.surface)
    }
}

impl Camera<f32, f32> for PanoramaCamera {
    fn get_ray(&self, _sampler: &mut Box<dyn Sampler>, _lambda: f32, u: f32, v: f32) -> (Ray, f32) {
        // spherical aperture, though of zero size at the moment. see TODO in struct definition
        // let offset: Vec3 = self.radius * random_in_unit_sphere(sampler.draw_2d());

        let ray_origin = self.origin;

        let ray_direction = {
            let (angle_x, angle_y) = (self.angle_span.0 * (u - 0.5), self.angle_span.1 * (0.5 - v));
            // angle y is elevation from horizon
            // angle x is azimuthal angle from the center line
            let (sin_x, cos_x) = angle_x.sin_cos();
            let (sin_y, cos_y) = angle_y.sin_cos();
            let vec = Vec3::new(sin_x * cos_y, sin_y, cos_x * cos_y);
            // trace!("{:?}", vec);
            self.transform.to_world(vec)
        };

        debug_assert!(ray_origin.is_finite());
        debug_assert!(ray_direction.is_finite());
        // TODO: determine whether this pdf needs to be set, and to what.
        // probably self.angle_span.0 * self.angle_span.1 / (4 pi^2 * untransformed_vec.y()) or something like that, mirroring the jacobian in the environment map code

        (Ray::new(ray_origin, ray_direction), 1.0)
    }
    fn with_aspect_ratio(self, _aspect_ratio: f32) -> Self {
        self
    }
    // returns None if the point on the lens was not from a valid pixel
    fn get_pixel_for_ray(&self, _ray: Ray, _lambda: f32) -> Option<(f32, f32)> {
        // TODO
        todo!()
    }

    fn eval_we(
        &self,
        _lambda: f32,
        _normal: Vec3,
        _from: Point3,
        _to: Point3,
    ) -> (f32, PDF<f32, SolidAngle>) {
        // pdf is projected solid angle wrt `to` point unless (from - self.origin).norm_squared() > self.radius * self.radius
        // if radius is very small, then the projected solid angle becomes very small
        // becomes a delta distribution
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
        (ray, ray.direction, tau.into())
    }
}

unsafe impl Send for PanoramaCamera {}
unsafe impl Sync for PanoramaCamera {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_panorama_camera() {
        let panorama = PanoramaCamera::new(
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Vec3::Z,
            180.0,
            90.0,
        );
    }
}
