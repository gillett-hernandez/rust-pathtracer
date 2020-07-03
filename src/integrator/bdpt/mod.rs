mod utils;

use utils::*;

// use crate::aabb::HasBoundingBox;
use crate::config::RenderSettings;
use crate::hittable::Hittable;
use crate::integrator::{CameraId, GenericIntegrator, Sample};
use crate::material::Material;
use crate::materials::*;
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use crate::world::World;
use crate::TransportMode;

// use std::f32::INFINITY;
use std::sync::Arc;

pub struct BDPTIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub wavelength_bounds: Bounds1D,
}

impl GenericIntegrator for BDPTIntegrator {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        settings: &RenderSettings,
        camera_sample: (Ray, CameraId),
        samples: &mut Vec<(Sample, CameraId)>,
    ) -> SingleWavelength {
        // setup: decide light, emit ray from light, decide camera, emit ray from camera, connect light path vertices to camera path vertices.

        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();
        // let camera_pick = sampler.draw_1d();
        // let (camera, camera_id, camera_pick_pdf) = self
        //     .world
        //     .pick_random_camera(camera_pick)
        //     .expect("camera pick failed");
        // let camera_ray;
        // if let Some(camera_surface) = camera.get_surface() {
        //     // let (direction, camera_pdf) = camera_surface.sample(camera_direction_sample, hit.point);
        //     // let direction = direction.normalized();
        //     for _ in 0..self.max_lens_sample_attempts {
        //         let film_sample = sampler.draw_2d();
        //         let lens_sample = sampler.draw_2d(); // sometimes called aperture sample
        //         let (point_on_lens, lens_normal, pdf) = camera.sample_we(film_sample, lens_sample);
        //         let camera_pdf = pdf * camera_pick_pdf;
        //         if camera_pdf.0 >= 0.0 {
        //             camera_ray = Ray::new(point_on_lens, lens_normal);
        //             break;
        //         }
        //     }
        // } else {
        //     return;
        // }
        let camera_ray = camera_sample.0;
        let camera_id = camera_sample.1;

        let env_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;

        let start_light_vertex;
        if light_pick_sample.x >= env_sampling_probability {
            light_pick_sample.x = ((light_pick_sample.x - env_sampling_probability)
                / (1.0 - env_sampling_probability))
                .clamp(0.0, 1.0);

            if self.world.lights.len() == 0 {
                return SingleWavelength::BLACK;
            }
            let (light, light_pick_pdf) = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal, area_pdf) =
                light.sample_surface(sampler.draw_2d());

            let mat_id = light.get_material_id();
            let material = self.world.get_material(mat_id);
            // println!("sampled light emission in instance light branch");
            let maybe_sampled = material.sample_emission(
                light_surface_point,
                light_surface_normal,
                VISIBLE_RANGE,
                sampler.draw_2d(),
                wavelength_sample,
            );
            sampled = if let Some(data) = maybe_sampled {
                data
            } else {
                println!("light instance is {:?}, material is {:?}", light, material);
                panic!();
            };

            let directional_pdf = sampled.2;
            // if delta light, the pdf_forward is only directional_pdf
            let pdf_forward: PDF =
                directional_pdf / (light_surface_normal * (&sampled.0).direction).abs();
            let pdf_backward: PDF = light_pick_pdf * area_pdf;
            debug_assert!(
                pdf_forward.0.is_finite(),
                "pdf_forward was not finite {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                pdf_forward,  // NaN
                pdf_backward, // 0.494
                sampled.0,
                material,
                directional_pdf, // NaN
                light_surface_point,
                light_surface_normal, // -Z
                sampled.1.energy      // 9.88
            );
            debug_assert!(
                pdf_backward.0.is_finite(),
                "pdf_backward was not finite {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                pdf_backward,
                pdf_forward,
                material,
                directional_pdf,
                light_surface_point,
                light_surface_normal,
                sampled.1.energy
            );

            start_light_vertex = Vertex::new(
                VertexType::LightSource(LightSourceType::Instance),
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
            // world
            light_pick_sample.x = (light_pick_sample.x / env_sampling_probability).clamp(0.0, 1.0);
            // sample world env
            let world_radius = self.world.get_world_radius();
            sampled = self.world.environment.sample_emission(
                world_radius,
                sampler.draw_2d(),
                sampler.draw_2d(),
                VISIBLE_RANGE,
                wavelength_sample,
            );
            let light_g_term = 1.0;
            let directional_pdf = sampled.2;
            start_light_vertex = Vertex::new(
                VertexType::LightSource(LightSourceType::Environment),
                0.0,
                sampled.1.lambda,
                //TODO: Fix the env sampling in eval_unweighted_contribution so that it picks a better env vertex
                sampled.0.origin, // + -1000.0 * sampled.0.direction, // push back env sample very far. temporary hack
                sampled.0.direction,
                MaterialId::Light(0),
                0,
                sampled.1.energy,
                directional_pdf.0,
                1.0,
                light_g_term,
            );
        };

        let light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;

        // idea: do limited branching and store vertices in a tree format that easily allows for traversal and connections
        let mut light_path: Vec<Vertex> = Vec::with_capacity(1 + self.max_bounces as usize);
        let mut eye_path: Vec<Vertex> = Vec::with_capacity(1 + self.max_bounces as usize);

        eye_path.push(Vertex::new(
            VertexType::Camera,
            camera_ray.time,
            lambda,
            camera_ray.origin,
            camera_ray.direction,
            MaterialId::Camera(camera_id),
            0,
            SingleEnergy::ONE,
            1.0,
            0.01,
            1.0,
        ));
        light_path.push(start_light_vertex);

        let _additional_contribution_eye_path = random_walk(
            camera_ray,
            lambda,
            self.max_bounces,
            SingleEnergy::ONE,
            TransportMode::Importance,
            sampler,
            &self.world,
            &mut eye_path,
            3,
        );
        random_walk(
            light_ray,
            lambda,
            self.max_bounces,
            radiance,
            TransportMode::Radiance,
            sampler,
            &self.world,
            &mut light_path,
            3,
        );

        for vertex in eye_path.iter() {
            debug_assert!(vertex.pdf_forward.is_finite(), "{:?}", eye_path);
            debug_assert!(vertex.pdf_backward.is_finite(), "{:?}", eye_path);
            debug_assert!(vertex.veach_g.is_finite(), "{:?}", eye_path);
            debug_assert!(vertex.point.0.is_finite().all(), "{:?}", eye_path);
            debug_assert!(vertex.normal.0.is_finite().all(), "{:?}", eye_path);
        }

        for vertex in light_path.iter() {
            debug_assert!(vertex.pdf_forward.is_finite(), "{:?}", light_path);
            debug_assert!(vertex.pdf_backward.is_finite(), "{:?}", light_path);
            debug_assert!(vertex.veach_g.is_finite(), "{:?}", light_path);
            debug_assert!(vertex.point.0.is_finite().all(), "{:?}", light_path);
            debug_assert!(vertex.normal.0.is_finite().all(), "{:?}", light_path);
        }

        let (eye_vertex_count, light_vertex_count) = (eye_path.len(), light_path.len());

        static MIS_ENABLED: bool = true;
        let russian_roulette_threshold = 0.005;
        if let Some((s, t)) = settings.selected_pair {
            if s <= light_vertex_count && t <= eye_vertex_count {
                let res = eval_unweighted_contribution(
                    &self.world,
                    &light_path,
                    s,
                    &eye_path,
                    t,
                    sampler,
                    russian_roulette_threshold,
                );

                match res {
                    SampleKind::Sampled((factor, g)) => {
                        if g == 0.0 || factor == SingleEnergy::ZERO {
                            return SingleWavelength::BLACK;
                        }
                        let weight = if MIS_ENABLED {
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
                            1.0 / ((s + t) as f32)
                        };
                        return SingleWavelength::new(lambda, weight * factor);
                    }

                    SampleKind::Splatted((factor, g)) => {
                        // println!("should be splatting0 {:?} and {}", factor, g);
                        if g == 0.0 || factor == SingleEnergy::ZERO {
                            return SingleWavelength::BLACK;
                        }
                        let weight = if MIS_ENABLED {
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
                            1.0 / ((s + t) as f32)
                        };
                        let contribution = weight * factor;
                        let last_light_vertex = light_path[s - 1];
                        let (vert_on_lens, vert_in_scene) = if t == 1 {
                            // t = 1 case
                            // light path hit somewhere in the scene, and is being connected to the vertex on the camera lens.
                            // in which case, the vertex on the camera lens is eye_path[0] and the vertex in the scene is light_path[s-1]
                            (eye_path[0], last_light_vertex)
                        } else if t == 0 {
                            // t = 0 case, light path directly intersected camera lens element, and thus the point on the camera is last_light_vertex
                            // and the other point that determines the ray is light_path[s-2];
                            (last_light_vertex, light_path[s - 2])
                        } else {
                            panic!()
                        };
                        if let MaterialId::Camera(camera_id) = vert_on_lens.material_id {
                            let camera = self.world.get_camera(camera_id as usize);
                            let ray = Ray::new(
                                vert_on_lens.point,
                                (vert_in_scene.point - vert_on_lens.point).normalized(),
                            );

                            if let Some(pixel_uv) = camera.get_pixel_for_ray(ray) {
                                // println!("found good pixel uv at {:?}", pixel_uv);
                                let sample = Sample::LightSample(
                                    SingleWavelength::new(lambda, contribution),
                                    pixel_uv,
                                );
                                samples.push((sample, camera_id));
                            } else {
                                // println!("pixel uv was nothing");
                            }
                        }
                        return SingleWavelength::BLACK;
                    }
                }
            }
            return SingleWavelength::BLACK;
        }

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

                // let mut g = 1.0;
                let result = eval_unweighted_contribution(
                    &self.world,
                    &light_path,
                    s,
                    &eye_path,
                    t,
                    sampler,
                    russian_roulette_threshold,
                );
                let (factor, g, calculate_splat) = match result {
                    SampleKind::Sampled((factor, g)) => (factor, g, false),
                    SampleKind::Splatted((factor, g)) => (factor, g, true),
                };
                if factor == SingleEnergy::ZERO {
                    continue;
                }
                if g == 0.0 {
                    continue;
                }
                let weight = if MIS_ENABLED {
                    eval_mis(
                        &self.world,
                        &light_path,
                        s,
                        &eye_path,
                        t,
                        g,
                        |weights: &Vec<f32>| -> f32 {
                            1.0 / weights.iter().map(|&v| v * v).sum::<f32>()
                        },
                    )
                } else {
                    1.0 / ((s + t) as f32)
                };
                if weight == 0.0 {
                    continue;
                }
                if calculate_splat {
                    // println!("should be splatting2");
                    let contribution = weight * factor;
                    let last_light_vertex = light_path[s - 1];
                    let (vert_on_lens, vert_in_scene) = if t == 1 {
                        // t = 1 case
                        // light path hit somewhere in the scene, and is being connected to the vertex on the camera lens.
                        // in which case, the vertex on the camera lens is eye_path[0] and the vertex in the scene is light_path[s-1]
                        (eye_path[0], last_light_vertex)
                    } else if t == 0 {
                        // t = 0 case, light path directly intersected camera lens element, and thus the point on the camera is last_light_vertex
                        // and the other point that determines the ray is light_path[s-2];
                        (last_light_vertex, light_path[s - 2])
                    } else {
                        panic!()
                    };
                    if let MaterialId::Camera(camera_id) = vert_on_lens.material_id {
                        let camera = self.world.get_camera(camera_id as usize);
                        let ray = Ray::new(
                            vert_on_lens.point,
                            (vert_in_scene.point - vert_on_lens.point).normalized(),
                        );

                        if let Some(pixel_uv) = camera.get_pixel_for_ray(ray) {
                            // println!("found good pixel uv at {:?}", pixel_uv);
                            let sample = Sample::LightSample(
                                SingleWavelength::new(lambda, contribution),
                                pixel_uv,
                            );
                            samples.push((sample, camera_id));
                        } else {
                            // println!("pixel uv was nothing");
                        }
                    }
                } else {
                    sum += weight * factor;
                    debug_assert!(sum.0.is_finite(), "{:?} {:?}", weight, factor);
                }
            }
        }
        debug_assert!(sum.0.is_finite());
        SingleWavelength::new(lambda, sum)
    }
}
