pub mod utils;

use utils::*;

// use crate::aabb::HasBoundingBox;
use crate::config::RenderSettings;
use crate::hittable::Hittable;
use crate::integrator::utils::*;
use crate::integrator::*;
use crate::materials::*;
use crate::math::*;
use crate::world::TransportMode;
use crate::world::World;
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
        camera_sample: ((f32, f32), CameraId),
        _sample_id: usize,
        samples: &mut Vec<(Sample, CameraId)>,
        mut profile: &mut Profile,
    ) -> XYZColor {
        // setup: decide light, emit ray from light, decide camera, emit ray from camera, connect light path vertices to camera path vertices.

        let wavelength_sample = sampler.draw_1d();
        let light_pick_sample = sampler.draw_1d();
        let env_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;

        let start_light_vertex;
        let (light_pick_sample, sample_env) =
            light_pick_sample.choose(env_sampling_probability, true, false);
        if !sample_env {
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
                self.wavelength_bounds,
                sampler.draw_2d(),
                wavelength_sample,
            );
            sampled = if let Some(data) = maybe_sampled {
                data
            } else {
                println!("failed to sample, material is {:?}", material.get_name());
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
                pdf_forward,
                pdf_backward,
                sampled.0,
                material.get_name(),
                directional_pdf,
                light_surface_point,
                light_surface_normal,
                sampled.1.energy
            );
            debug_assert!(
                pdf_backward.0.is_finite(),
                "pdf_backward was not finite {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                pdf_backward,
                pdf_forward,
                material.get_name(),
                directional_pdf,
                light_surface_point,
                light_surface_normal,
                sampled.1.energy
            );

            start_light_vertex = Vertex::new(
                VertexType::LightSource(LightSourceType::Instance),
                0.0,
                sampled.1.lambda,
                Vec3::ZERO,
                light_surface_point,
                light_surface_normal,
                (0.0, 0.0),
                mat_id,
                light.get_instance_id(),
                sampled.1.energy,
                pdf_forward.into(),
                pdf_backward.into(),
                1.0,
            );
        } else {
            // sample world env
            let world_radius = self.world.get_world_radius();
            let world_center = self.world.get_center();
            sampled = self.world.environment.sample_emission(
                world_radius,
                world_center,
                sampler.draw_2d(),
                sampler.draw_2d(),
                self.wavelength_bounds,
                wavelength_sample,
            );
            let light_g_term = 1.0;
            let directional_pdf = sampled.2;
            start_light_vertex = Vertex::new(
                VertexType::LightSource(LightSourceType::Environment),
                0.0,
                sampled.1.lambda,
                Vec3::ZERO,
                sampled.0.origin,
                -sampled.0.direction,
                (0.0, 0.0),
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
        assert!(
            (sampled.3).0 > 0.0,
            "{:?} {:?} {:?} {:?}",
            sampled.0,
            sampled.1,
            sampled.2,
            sampled.3
        );

        let camera_ray;
        let camera_id = camera_sample.1;
        let camera = self.world.get_camera(camera_id);
        let film_sample = Sample2D::new(
            (camera_sample.0).0.clamp(0.0, 1.0 - std::f32::EPSILON),
            (camera_sample.0).1.clamp(0.0, 1.0 - std::f32::EPSILON),
        );
        let aperture_sample = sampler.draw_2d(); // sometimes called aperture sample
        let (sampled_camera_ray, lens_normal, camera_pdf) =
            camera.sample_we(film_sample, aperture_sample, lambda);
        // let camera_pdf = pdf;
        camera_ray = sampled_camera_ray;

        let radiance = sampled.1.energy;

        // idea: do limited branching and store vertices in a tree format that easily allows for traversal and connections
        let mut light_path: Vec<Vertex> = Vec::with_capacity(1 + self.max_bounces as usize);
        let mut eye_path: Vec<Vertex> = Vec::with_capacity(1 + self.max_bounces as usize);

        eye_path.push(Vertex::new(
            VertexType::Camera,
            camera_ray.time,
            lambda,
            Vec3::ZERO,
            camera_ray.origin,
            lens_normal,
            (0.0, 0.0),
            MaterialId::Camera(camera_id as u16),
            0,
            SingleEnergy::ONE,
            camera_pdf.0,
            0.0,
            1.0,
        ));
        light_path.push(start_light_vertex);
        let (sp1, tp1) = if let IntegratorKind::BDPT {
            selected_pair: Some((s, t)),
        } = settings.integrator
        {
            (s + 1, t + 1)
        } else {
            (self.max_bounces as usize, self.max_bounces as usize)
        };

        let _additional_contribution_eye_path = random_walk(
            camera_ray,
            lambda,
            tp1 as u16,
            SingleEnergy::ONE,
            TransportMode::Importance,
            sampler,
            &self.world,
            &mut eye_path,
            settings.min_bounces.unwrap_or(3),
            &mut profile,
        );
        random_walk(
            light_ray,
            lambda,
            sp1 as u16,
            radiance,
            TransportMode::Radiance,
            sampler,
            &self.world,
            &mut light_path,
            settings.min_bounces.unwrap_or(3),
            &mut profile,
        );

        profile.camera_rays += 1;
        profile.light_rays += 1;

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
        if let IntegratorKind::BDPT {
            selected_pair: Some((s, t)),
        } = settings.integrator
        {
            if s <= light_vertex_count && t <= eye_vertex_count {
                let res = eval_unweighted_contribution(
                    &self.world,
                    &light_path,
                    s,
                    &eye_path,
                    t,
                    sampler,
                    russian_roulette_threshold,
                    &mut profile,
                );

                match res {
                    SampleKind::Sampled((factor, g)) => {
                        if g == 0.0 || factor == SingleEnergy::ZERO {
                            return XYZColor::from(SingleWavelength::BLACK);
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
                        } / (sampled.3).0;
                        return XYZColor::from(SingleWavelength::new(
                            lambda,
                            weight * factor / (sampled.3).0,
                        ));
                    }

                    SampleKind::Splatted((factor, g)) => {
                        // println!("should be splatting0 {:?} and {}", factor, g);
                        if g == 0.0 || factor == SingleEnergy::ZERO {
                            return XYZColor::from(SingleWavelength::BLACK);
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
                        } / (sampled.3).0;
                        let contribution = weight * factor / (sampled.3).0;
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
                                    XYZColor::from(SingleWavelength::new(lambda, contribution)),
                                    pixel_uv,
                                );
                                samples.push((sample, camera_id as usize));
                            } else {
                                // println!("pixel uv was nothing");
                            }
                        }
                        return XYZColor::from(SingleWavelength::BLACK);
                    }
                }
            }
            return XYZColor::from(SingleWavelength::BLACK);
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
                    &mut profile,
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
                } / (sampled.3).0;
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
                                XYZColor::from(SingleWavelength::new(lambda, contribution)),
                                pixel_uv,
                            );
                            samples.push((sample, camera_id as usize));
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
        XYZColor::from(SingleWavelength::new(lambda, sum))
    }
}
