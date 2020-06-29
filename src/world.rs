// use crate::camera::BundledCamera;
use crate::camera::Camera;
use crate::hittable::*;
use crate::materials::MaterialTable;
use crate::math::*;

pub use crate::accelerator::{Accelerator, AcceleratorType};
pub use crate::geometry::Instance;
pub use crate::materials::*;

#[derive(Clone)]
pub struct EnvironmentMap {
    pub color: SPD,
    pub strength: f32,
}

impl EnvironmentMap {
    pub const fn new(color: SPD, strength: f32) -> Self {
        EnvironmentMap { color, strength }
    }
    pub fn sample_spd(
        &self,
        _uv: (f32, f32),
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> Option<(SingleWavelength, PDF)> {
        // sample emission at a given uv when wavelength is yet to be determined.
        // used when a camera ray hits an environment map without ever having been assigned a wavelength.

        // later use uv for texture accessing
        let (mut sw, pdf) = self.color.sample_power_and_pdf(wavelength_range, sample);
        sw.energy *= self.strength;
        Some((sw, pdf))
    }

    pub fn emission(&self, uv: (f32, f32), lambda: f32) -> SingleEnergy {
        // evaluate emission at uv coordinate and wavelength

        let phi = PI * (2.0 * uv.0 - 1.0);
        let (y, x) = phi.sin_cos();
        let z = (PI * uv.1).cos();
        debug_assert!(lambda > 0.0);
        // SingleEnergy::new(self.color.evaluate_power(lambda))
        SingleEnergy::new(self.color.evaluate_power(lambda) * self.strength)
    }

    pub fn sample_emission(
        &self,
        world_radius: f32,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength, PDF) {
        // sample env map cdf to get light ray, based on env map strength
        let (mut sw, pdf) = self
            .color
            .sample_power_and_pdf(wavelength_range, wavelength_sample);
        sw.energy *= self.strength;
        let random_direction = random_on_unit_sphere(direction_sample);
        let random_on_world = world_radius * random_on_unit_sphere(position_sample);
        let point = Point3::from(random_on_world);

        (Ray::new(point, random_direction), sw, pdf)
    }
}

impl From<EnvironmentMap> for DiffuseLight {
    fn from(map: EnvironmentMap) -> Self {
        DiffuseLight::new(map.color, Sidedness::Dual)
    }
}

#[derive(Clone)]
pub struct World {
    pub accelerator: Accelerator,
    pub lights: Vec<usize>,
    pub cameras: Vec<Camera>,
    pub materials: MaterialTable,
    pub environment: EnvironmentMap,
    env_sampling_probability: f32,
}

impl World {
    pub fn new(
        instances: Vec<Instance>,
        materials: MaterialTable,
        environment: EnvironmentMap,
        env_sampling_probability: f32,
    ) -> Self {
        let mut lights = Vec::new();
        for instance in instances.iter() {
            if let MaterialId::Light(id) = instance.get_material_id() {
                println!(
                    "adding light with mat id {:?} and instance id {:?} to lights list",
                    id, instance.instance_id
                );
                lights.push(instance.instance_id as usize);
            }
        }
        let accelerator = Accelerator::new(instances, AcceleratorType::List);
        World {
            accelerator,
            lights,
            cameras: Vec::new(),
            materials,
            environment,
            env_sampling_probability,
        }
    }
    pub fn pick_random_light(&self, s: Sample1D) -> Option<(&Instance, PDF)> {
        // currently just uniform sampling
        let length = self.lights.len();
        if length == 0 {
            None
        } else {
            let x = s.x;
            let idx = (length as f32 * x).clamp(0.0, length as f32 - 1.0) as usize;
            debug_assert!(
                idx < self.lights.len(),
                "{}, {}, {}, {}",
                x,
                length as f32 * x,
                idx,
                length as usize
            );
            Some((
                self.accelerator.get_primitive(self.lights[idx]),
                PDF::from(1.0 / length as f32),
            ))
        }
    }

    pub fn pick_random_camera(&self, s: Sample1D) -> Option<(&Camera, usize, PDF)> {
        // currently just uniform sampling
        let length = self.cameras.len();
        if length == 0 {
            None
        } else {
            let x = s.x;
            let idx = (length as f32 * x).clamp(0.0, length as f32 - 1.0) as usize;
            debug_assert!(
                idx < self.lights.len(),
                "{}, {}, {}, {}",
                x,
                length as f32 * x,
                idx,
                length as usize
            );
            Some((self.get_camera(idx), idx, PDF::from(1.0 / length as f32)))
        }
    }

    pub fn instance_is_light(&self, instance_id: usize) -> bool {
        self.lights.contains(&instance_id)
    }

    pub fn get_material(&self, mat_id: MaterialId) -> &MaterialEnum {
        let id: usize = mat_id.into();
        &self.materials[id]
    }

    pub fn get_primitive(&self, index: usize) -> &Instance {
        self.accelerator.get_primitive(index)
    }

    pub fn get_camera(&self, index: usize) -> &Camera {
        &self.cameras[index]
    }

    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        self.accelerator.hit(r, t0, t1)
    }

    pub fn get_env_sampling_probability(&self) -> f32 {
        if self.lights.len() > 0 {
            self.env_sampling_probability
        } else {
            1.0
        }
    }

    pub fn assign_cameras(&mut self, cameras: Vec<Camera>, add_and_rebuild_scene: bool) {
        // reconfigures the scene's cameras and rebuilds the scene accelerator if specified
        if add_and_rebuild_scene {
            for camera in self.cameras.iter() {
                let instances = &mut self.accelerator.instances;
                if let Some(camera_surface) = camera.get_surface() {
                    println!("removing camera surface {:?}", &camera_surface);
                    instances.remove_item(&camera_surface);
                }
            }
        }

        self.cameras = cameras;

        if add_and_rebuild_scene {
            let instances = &mut self.accelerator.instances;
            for (cam_id, cam) in self.cameras.iter().enumerate() {
                if let Some(mut surface) = cam.get_surface() {
                    let id = instances.len();
                    surface.instance_id = id;
                    surface.material_id = MaterialId::Camera(cam_id as u8);
                    println!("adding camera {:?} with id {}", &surface, cam_id);
                    instances.push(surface);
                }
            }
            self.accelerator.rebuild();
        }
    }
}

impl HasBoundingBox for World {
    fn bounding_box(&self) -> AABB {
        self.accelerator.bounding_box()
    }
}
