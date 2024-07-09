mod environment;
mod importance_map;

pub use environment::EnvironmentMap;
pub use importance_map::ImportanceMap;

use crate::prelude::*;

use crate::hittable::*;

use crate::mediums::MediumTable;

pub use crate::accelerator::{Accelerator, AcceleratorType};
pub use crate::geometry::*;
pub use crate::materials::*;

#[derive(Clone, Debug)]
pub struct World {
    pub accelerator: Accelerator,
    pub lights: Vec<InstanceId>,
    pub cameras: Vec<CameraEnum>,
    pub materials: MaterialTable,
    pub mediums: MediumTable,
    pub environment: EnvironmentMap,
    env_sampling_probability: f32,
    pub radius: f32,
    pub center: Point3,
}

impl World {
    pub fn new(
        instances: Vec<Instance>,
        materials: MaterialTable,
        mediums: MediumTable,
        environment: EnvironmentMap,
        cameras: Vec<CameraEnum>,
        mut env_sampling_probability: f32,
        accelerator_type: AcceleratorType,
    ) -> Self {
        // TODO: add light accelerator data structure, to prevent sampling very distant (and small) lights when there are lights closer by

        let mut lights = Vec::new();
        for instance in instances.iter() {
            match &instance.aggregate {
                Aggregate::Mesh(mesh) => {
                    for tri in mesh.triangles.as_ref().unwrap() {
                        if let MaterialId::Light(id) = tri.get_material_id() {
                            info!(
                            "adding light with mat id Light({:?}) and instance id {:?} to lights list",
                            id, instance.instance_id
                        );
                            lights.push(instance.instance_id as InstanceId);
                        }
                    }
                }
                _ => {
                    if let MaterialId::Light(id) = instance.get_material_id() {
                        info!(
                        "adding light with mat id Light({:?}) and instance id {:?} to lights list",
                        id, instance.instance_id
                    );
                        lights.push(instance.instance_id as InstanceId);
                    }
                }
            }
        }
        let accelerator = Accelerator::new(instances, accelerator_type);

        let world_aabb = accelerator.aabb();
        let span = world_aabb.max - world_aabb.min;
        let center: Point3 = world_aabb.min + span / 2.0;
        let radius = span.norm() / 2.0;
        info!(
            "world radius is {:?} meters, world center is at {:?}",
            radius, center
        );
        if lights.is_empty() {
            warn!("the world had no lights, so force-setting env_sampling_probability to 1.0");
            env_sampling_probability = 1.0;
        }
        let world = World {
            accelerator,
            lights,
            cameras,
            materials,
            mediums,
            environment,
            env_sampling_probability,
            radius,
            center,
        };
        if env_sampling_probability == 1.0 || env_sampling_probability == 0.0 {
            warn!(
                "env sampling probability is at an extrema of {}",
                env_sampling_probability
            );
        }
        world
    }
    pub fn pick_random_light(&self, s: Sample1D) -> Option<(&Instance, PDF<f32, Uniform01>)> {
        // currently just uniform sampling
        // TODO: change method to take into account the location from which the light is being picked, to allow light trees or other heuristics
        // i.e. a projected solid angle * power heuristic and pdf
        // maybe use reservoir sampling?
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
                length
            );
            Some((
                self.accelerator.get_primitive(self.lights[idx]),
                PDF::from(1.0 / length as f32),
            ))
        }
    }

    pub fn pick_random_camera(
        &self,
        s: Sample1D,
    ) -> Option<(&CameraEnum, usize, PDF<f32, Uniform01>)> {
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
                length
            );
            Some((self.get_camera(idx), idx, PDF::from(1.0 / length as f32)))
        }
    }

    pub fn instance_is_light(&self, instance_id: InstanceId) -> bool {
        self.lights.contains(&instance_id)
    }

    pub fn get_material(&self, mat_id: MaterialId) -> &MaterialEnum {
        let id: usize = mat_id.into();
        &self.materials[id]
    }

    pub fn get_primitive(&self, index: InstanceId) -> &Instance {
        self.accelerator.get_primitive(index)
    }

    pub fn get_camera(&self, index: usize) -> &CameraEnum {
        &self.cameras[index]
    }

    pub fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        self.accelerator.hit(r, t0, t1)
    }

    pub fn get_env_sampling_probability(&self) -> f32 {
        if !self.lights.is_empty() {
            self.env_sampling_probability
        } else {
            1.0
        }
    }

    #[deprecated]
    pub fn assign_cameras(&mut self, _: Vec<CameraEnum>, _: bool) {
        unimplemented!()
        // // reconfigures the scene's cameras and rebuilds the scene accelerator if specified
        // if add_and_rebuild_scene {
        //     match &mut self.accelerator {
        //         Accelerator::List { ref mut instances } => {
        //             for camera in self.cameras.iter() {
        //                 if let Some(camera_surface) = camera.get_surface() {
        //                     println!("removing camera surface {:?}", &camera_surface);
        //                     // instances.remove_item(&camera_surface);
        //                     let maybe_id = instances.binary_search(camera_surface);
        //                     if let Ok(id) = maybe_id {
        //                         instances.remove(id);
        //                     }
        //                 }
        //             }
        //         }
        //         Accelerator::BVH {
        //             ref mut instances,
        //             bvh: _,
        //         } => {
        //             for camera in self.cameras.iter() {
        //                 if let Some(camera_surface) = camera.get_surface() {
        //                     println!("removing camera surface {:?}", &camera_surface);
        //                     // instances.remove_item(&camera_surface);
        //                     let maybe_id = instances.binary_search(camera_surface);
        //                     if let Ok(id) = maybe_id {
        //                         instances.remove(id);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // self.cameras = cameras;

        // if add_and_rebuild_scene {
        //     match &mut self.accelerator {
        //         Accelerator::List { ref mut instances } => {
        //             for (cam_id, cam) in self.cameras.iter().enumerate() {
        //                 if let Some(surface) = cam.get_surface() {
        //                     let mut surface = surface.clone();
        //                     let id = instances.len() as InstanceId;
        //                     surface.instance_id = id;
        //                     surface.material_id = Some(MaterialId::Camera(cam_id as u16));
        //                     println!("adding camera {:?} with id {}", &surface, cam_id);
        //                     instances.push(surface);
        //                 }
        //             }
        //         }
        //         Accelerator::BVH {
        //             ref mut instances,
        //             bvh: _,
        //         } => {
        //             for (cam_id, cam) in self.cameras.iter().enumerate() {
        //                 if let Some(surface) = cam.get_surface() {
        //                     let mut surface = surface.clone();
        //                     let id = instances.len() as InstanceId;
        //                     surface.instance_id = id;
        //                     surface.material_id = Some(MaterialId::Camera(cam_id as u16));
        //                     println!("adding camera {:?} with id {}", &surface, cam_id);
        //                     instances.push(surface);
        //                 }
        //             }
        //         }
        //     }

        //     self.accelerator.rebuild();
        // }
    }
}

impl HasBoundingBox for World {
    fn aabb(&self) -> AABB {
        self.accelerator.aabb()
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::parsing::{config::Config, construct_world};

    use super::*;

    #[test]

    fn test_world_intersection() {
        crate::log_test_setup();
        let mut default_config = Config::load_default();
        let mut handles = Vec::new();
        let world = construct_world(
            &mut default_config,
            PathBuf::from("data/scenes/test_lighting_north.toml"),
            &mut handles,
        )
        .unwrap();
        for handle in handles {
            let _ = handle.join();
        }

        let ray = Ray::new(Point3::new(0.0, 0.0, 7.0), -Vec3::Z);

        let aabb = world.aabb();
        println!("{:?}", aabb);

        let maybe_hit = world.hit(ray, 0.0, f32::INFINITY);
        assert!(maybe_hit.is_some());
        let intersection = maybe_hit.unwrap();

        println!("{:?}", intersection);
    }
}
