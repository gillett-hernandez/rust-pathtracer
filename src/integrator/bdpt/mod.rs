mod helpers;

use helpers::*;

use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::camera::Camera;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;

// use std::f32::INFINITY;
use std::sync::Arc;

use crate::integrator::{CameraId, GenericIntegrator, Sample};

pub struct BDPTIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub specific_pair: Option<(usize, usize)>,
    pub cameras: Vec<Camera>,
}

impl GenericIntegrator for BDPTIntegrator {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        camera_ray: Ray,
        samples: &mut Vec<(Sample, CameraId)>,
    ) -> SingleWavelength {
        // setup: decide light, emit ray from light, emit ray from camera, connect light path vertices to camera path vertices.

        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();

        let scene_light_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;

        let start_light_vertex;
        if self.world.lights.len() > 0 && light_pick_sample.x < scene_light_sampling_probability {
            light_pick_sample.x =
                (light_pick_sample.x / scene_light_sampling_probability).clamp(0.0, 1.0);
            let (light, light_pick_pdf) = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal, area_pdf) =
                light.sample_surface(sampler.draw_2d());

            let mat_id = light.get_material_id();
            let material: &Box<dyn Material> = &self.world.get_material(mat_id);
            // println!("sampled light emission in instance light branch");
            sampled = material
                .sample_emission(
                    light_surface_point,
                    light_surface_normal,
                    VISIBLE_RANGE,
                    sampler.draw_2d(),
                    wavelength_sample,
                )
                .unwrap();
            let light_g_term = (light_surface_normal * (&sampled.0).direction).abs();

            let directional_pdf = sampled.2;
            // if delta light, the pdf_forward is only directional_pdf
            let pdf_forward: PDF = directional_pdf / light_g_term;
            let pdf_backward: PDF = light_pick_pdf * area_pdf;

            start_light_vertex = Vertex::new(
                Type::LightSource(Source::Instance),
                0.0,
                sampled.1.lambda,
                light_surface_point,
                light_surface_normal,
                mat_id,
                light.get_instance_id(),
                sampled.1.energy,
                pdf_forward.into(),
                pdf_backward.into(),
                1.0,
            );
        } else {
            light_pick_sample.x = ((light_pick_sample.x - scene_light_sampling_probability)
                / (1.0 - scene_light_sampling_probability))
                .clamp(0.0, 1.0);
            // sample world env
            let world_aabb = self.world.accelerator.bounding_box();
            let world_radius = (world_aabb.max - world_aabb.min).0.abs().max_element() / 2.0;
            // println!("sampled light emission in world light branch");
            sampled = self.world.environment.sample_emission(
                world_radius,
                sampler.draw_2d(),
                VISIBLE_RANGE,
                wavelength_sample,
            );
            start_light_vertex = Vertex::new(
                Type::LightSource(Source::Environment),
                0.0,
                sampled.1.lambda,
                sampled.0.origin,
                sampled.0.direction,
                0,
                0,
                sampled.1.energy,
                1.0,
                1.0,
                1.0,
            );
        };

        let light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;

        // idea: do limited branching and store vertices in a tree format that easily allows for traversal and connections
        let mut light_path: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize);
        let mut eye_path: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize);

        eye_path.push(Vertex::new(
            Type::Camera,
            camera_ray.time,
            lambda,
            camera_ray.origin,
            camera_ray.direction,
            0,
            0,
            SingleEnergy::ONE,
            1.0,
            0.0,
            1.0,
        ));
        eye_path.push(Vertex::new(
            Type::Camera,
            camera_ray.time,
            lambda,
            camera_ray.origin + 0.05 * camera_ray.direction,
            camera_ray.direction,
            0,
            0,
            SingleEnergy::ONE,
            1.0,
            1.0,
            1.0,
        ));
        light_path.push(start_light_vertex);

        let _additional_contribution = random_walk(
            camera_ray,
            lambda,
            self.max_bounces,
            SingleEnergy::ONE,
            Type::Eye,
            sampler,
            &self.world,
            &mut eye_path,
        );
        random_walk(
            light_ray,
            lambda,
            self.max_bounces,
            radiance,
            Type::Light,
            sampler,
            &self.world,
            &mut light_path,
        );

        let (eye_vertex_count, light_vertex_count) = (eye_path.len(), light_path.len());

        if let Some((s, t)) = self.specific_pair {
            if s <= light_vertex_count && t <= eye_vertex_count {
                let (factor, _g) =
                    eval_unweighted_contribution(&self.world, &light_path, s, &eye_path, t);
                return SingleWavelength::new(lambda, factor);
            }
        }

        let mis_enabled = false;
        // let mut mis_nodes: Vec<MISNode> = Vec::new();
        // if mis_enabled {
        //     // for _ in 0..(self.max_bounces + 1) {
        //     //     mis_nodes.push(MISNode::new(1.0, 1.0, false));
        //     // }
        //     weights.fill(MISNode::new(1.0, 1.0, false));
        // }
        let mut sum = SingleEnergy::ZERO;
        // sum += additional_contribution.unwrap_or(SingleEnergy::ZERO);
        for path_length in 1..(1 + self.max_bounces as usize) {
            let path_vertex_count = path_length + 1;
            for s in 0..(path_vertex_count as usize) {
                let t = path_vertex_count - s;
                if s > light_vertex_count || t > eye_vertex_count {
                    continue;
                }
                if (s == 0 && t < 2) || (t == 0 && s < 2) || (s + t) < 2 {
                    continue;
                }

                if t < 2 {
                    continue;
                }
                let mut g = 1.0;
                let (factor, new_g) =
                    eval_unweighted_contribution(&self.world, &light_path, s, &eye_path, t);
                if factor == SingleEnergy::ZERO {
                    continue;
                }
                if new_g > 0.0 {
                    g = new_g;
                }
                let weight = if mis_enabled {
                    eval_mis(
                        &self.world,
                        &light_path,
                        s,
                        &eye_path,
                        t,
                        g,
                        |weights: &Vec<f32>| -> f32 { 1.0 / weights.iter().sum::<f32>() },
                    )
                } else {
                    1.0
                };
                if weight == 0.0 {
                    continue;
                }
                sum += weight * factor;
            }
        }
        SingleWavelength::new(lambda, sum)
    }
}
