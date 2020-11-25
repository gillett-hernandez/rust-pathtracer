mod environment;

pub use environment::EnvironmentMap;

use crate::camera::Camera;
use crate::hittable::*;
use crate::materials::MaterialTable;
use crate::math::*;

pub use crate::accelerator::{Accelerator, AcceleratorType};
pub use crate::geometry::*;
pub use crate::materials::*;

use std::collections::HashMap;

pub const NORMAL_OFFSET: f32 = 0.00001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TransportMode {
    Radiance,
    Importance,
}

impl Default for TransportMode {
    fn default() -> Self {
        TransportMode::Importance
    }
}

#[derive(Clone)]
pub struct World {
    pub accelerator: Accelerator,
    pub lights: Vec<usize>,
    pub cameras: HashMap<String, Camera>,
    pub materials: MaterialTable,
    pub environment: EnvironmentMap,
    env_sampling_probability: f32,
    radius: f32,
    center: Point3,
}

impl World {
    pub fn new(
        instances: Vec<Instance>,
        materials: MaterialTable,
        environment: EnvironmentMap,
        mut env_sampling_probability: f32,
        accelerator_type: AcceleratorType,
    ) -> Self {
        let mut lights = Vec::new();
        for instance in instances.iter() {
            match &instance.aggregate {
                Aggregate::Mesh(mesh) => {
                    for tri in (&mesh).triangles.as_ref().unwrap() {
                        if let MaterialId::Light(id) = tri.get_material_id() {
                            println!(
                            "adding light with mat id Light({:?}) and instance id {:?} to lights list",
                            id, instance.instance_id
                        );
                            lights.push(instance.instance_id as usize);
                        }
                    }
                }
                _ => {
                    if let MaterialId::Light(id) = instance.get_material_id() {
                        println!(
                        "adding light with mat id Light({:?}) and instance id {:?} to lights list",
                        id, instance.instance_id
                    );
                        lights.push(instance.instance_id as usize);
                    }
                }
            }
        }
        let accelerator = Accelerator::new(instances, accelerator_type);

        let world_aabb = accelerator.aabb();
        let span = world_aabb.max - world_aabb.min;
        let center: Point3 = world_aabb.min + span / 2.0;
        let radius = span.norm() / 2.0;
        println!(
            "world radius is {:?} meters, world center is at {:?}",
            radius, center
        );
        if lights.len() == 0 {
            println!("the world had no lights, so force-setting env_sampling_probability to 1.0");
            env_sampling_probability = 1.0;
        }
        let world = World {
            accelerator,
            lights,
            cameras: Vec::new(),
            materials,
            environment,
            env_sampling_probability,
            radius,
            center,
        };
        if env_sampling_probability == 1.0 || env_sampling_probability == 0.0 {
            println!(
                "warning! env sampling probability is at an extrema of {}",
                env_sampling_probability
            );
        }
        world
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
                idx < self.cameras.len(),
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

    pub fn get_camera(&self, index: String) -> &Camera {
        &self.cameras[&index]
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

    pub fn get_world_radius(&self) -> f32 {
        self.radius
    }

    pub fn get_center(&self) -> Point3 {
        self.center
    }

    pub fn assign_cameras(&mut self, cameras: Vec<Camera>, add_and_rebuild_scene: bool) {
        // reconfigures the scene's cameras and rebuilds the scene accelerator if specified
        if add_and_rebuild_scene {
            match &mut self.accelerator {
                Accelerator::List { ref mut instances } => {
                    for camera in self.cameras.iter() {
                        if let Some(camera_surface) = camera.get_surface() {
                            println!("removing camera surface {:?}", &camera_surface);
                            // instances.remove_item(&camera_surface);
                            let maybe_id = instances.binary_search(&camera_surface);
                            if let Ok(id) = maybe_id {
                                instances.remove(id);
                            }
                        }
                    }
                }
                Accelerator::BVH {
                    ref mut instances,
                    bvh: _,
                } => {
                    for camera in self.cameras.iter() {
                        if let Some(camera_surface) = camera.get_surface() {
                            println!("removing camera surface {:?}", &camera_surface);
                            // instances.remove_item(&camera_surface);
                            let maybe_id = instances.binary_search(&camera_surface);
                            if let Ok(id) = maybe_id {
                                instances.remove(id);
                            }
                        }
                    }
                }
            }
        }

        self.cameras = cameras;

        if add_and_rebuild_scene {
            match &mut self.accelerator {
                Accelerator::List { ref mut instances } => {
                    for (cam_id, cam) in self.cameras.iter().enumerate() {
                        if let Some(surface) = cam.get_surface() {
                            let mut surface = surface.clone();
                            let id = instances.len();
                            surface.instance_id = id;
                            surface.material_id = Some(MaterialId::Camera(cam_id as u16));
                            println!("adding camera {:?} with id {}", &surface, cam_id);
                            instances.push(surface);
                        }
                    }
                }
                Accelerator::BVH {
                    ref mut instances,
                    bvh: _,
                } => {
                    for (cam_id, cam) in self.cameras.iter().enumerate() {
                        if let Some(surface) = cam.get_surface() {
                            let mut surface = surface.clone();
                            let id = instances.len();
                            surface.instance_id = id;
                            surface.material_id = Some(MaterialId::Camera(cam_id as u16));
                            println!("adding camera {:?} with id {}", &surface, cam_id);
                            instances.push(surface);
                        }
                    }
                }
            }

            self.accelerator.rebuild();
        }
    }
}

impl HasBoundingBox for World {
    fn aabb(&self) -> AABB {
        self.accelerator.aabb()
    }
}
