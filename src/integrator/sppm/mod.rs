use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::*;
use crate::integrator::*;
use crate::materials::{Material, MaterialId};
use crate::math::*;
use crate::world::TransportMode;
use crate::world::World;
use math::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
// use crate::{INTERSECTION_TIME_OFFSET, NORMAL_OFFSET};

use std::{f32::EPSILON, sync::Arc};

use rayon::iter::ParallelIterator;
use rayon::prelude::*;

fn window_function(lambda1: f32, lambda2: f32) -> f32 {
    let diff = (lambda1 - lambda2).abs();
    1.0 / (1.0 + diff * diff)
}

pub struct PhotonMap {
    pub photons: Vec<SurfaceVertex>,
}

pub struct SPPMIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub russian_roulette: bool,
    pub camera_samples: u16,
    pub wavelength_bounds: Bounds1D,
    pub photon_map: Option<PhotonMap>,
    pub last_lambda: f32,
}

impl SamplerIntegrator for SPPMIntegrator {
    fn preprocess(
        &mut self,
        sampler: &mut Box<dyn Sampler>,
        _settings: &Vec<RenderSettings>,
        profile: &mut Profile,
    ) {
        let num_beams = 1000;

        let wavelength_sample = sampler.draw_1d();
        self.last_lambda = wavelength_sample.x;
        println!("preprocessing and mapping beams");
        let mut beams: Vec<Vec<SurfaceVertex>> = vec![Vec::new(); num_beams];
        let beams_profile = beams
            .par_iter_mut()
            .map(|beam| {
                let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                let mut profile = Profile::default();

                let env_sampling_probability = self.world.get_env_sampling_probability();

                let sampled;
                let start_light_vertex;
                loop {
                    let mut light_pick_sample = sampler.draw_1d();
                    if light_pick_sample.x >= env_sampling_probability {
                        light_pick_sample.x = ((light_pick_sample.x - env_sampling_probability)
                            / (1.0 - env_sampling_probability))
                            .clamp(0.0, 1.0);

                        if self.world.lights.len() == 0 {
                            continue;
                        }
                        let (light, light_pick_pdf) =
                            self.world.pick_random_light(light_pick_sample).unwrap();

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

                        start_light_vertex = SurfaceVertex::new(
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
                        break;
                    } else {
                        // world
                        light_pick_sample.x =
                            (light_pick_sample.x / env_sampling_probability).clamp(0.0, 1.0);
                        // sample world env
                        let world_radius = self.world.get_world_radius();
                        let world_center = self.world.get_center();
                        sampled = self.world.environment.sample_emission(
                            world_radius,
                            world_center,
                            sampler.draw_2d(),
                            sampler.draw_2d(),
                            VISIBLE_RANGE,
                            wavelength_sample,
                        );
                        let light_g_term = 1.0;
                        let directional_pdf = sampled.2;
                        start_light_vertex = SurfaceVertex::new(
                            VertexType::LightSource(LightSourceType::Environment),
                            0.0,
                            sampled.1.lambda,
                            Vec3::ZERO,
                            sampled.0.origin,
                            sampled.0.direction,
                            (0.0, 0.0),
                            MaterialId::Light(0),
                            0,
                            sampled.1.energy,
                            directional_pdf.0,
                            1.0,
                            light_g_term,
                        );
                        break;
                    };
                }

                // println!("{:?}", start_light_vertex);

                profile.light_rays += 1;
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
                let radiance = sampled.1.energy;

                *beam = Vec::with_capacity(self.max_bounces as usize);
                beam.push(start_light_vertex);
                let _ = random_walk(
                    light_ray,
                    lambda,
                    self.max_bounces,
                    radiance,
                    TransportMode::Radiance,
                    &mut sampler,
                    &self.world,
                    beam,
                    4,
                    &mut profile,
                );
                profile
            })
            .reduce(|| Profile::default(), |a, b| a.combine(b));
        *profile = profile.combine(beams_profile);
        self.photon_map = Some(PhotonMap {
            photons: beams.into_iter().flatten().collect(),
        });
        println!(
            "stored {} photons in the photon map",
            self.photon_map
                .as_ref()
                .map(|e| e.photons.len())
                .unwrap_or(0)
        );
    }
    fn color(
        &self,
        mut sampler: &mut Box<dyn Sampler>,
        camera_sample: ((f32, f32), CameraId),
        sample_id: usize,
        mut profile: &mut Profile,
    ) -> XYZColor {
        // naive implementation of SPPM
        // iterate through all deposited photons and add contributions based on if they are close to the eye vertex in question

        let camera_id = camera_sample.1;
        let camera = self.world.get_camera(camera_id as usize);

        let mut sum = XYZColor::ZERO;
        // let (direction, camera_pdf) = camera_surface.sample(camera_direction_sample, hit.point);
        // let direction = direction.normalized();
        let film_sample = Sample2D::new(
            (camera_sample.0).0.clamp(0.0, 1.0 - EPSILON),
            (camera_sample.0).1.clamp(0.0, 1.0 - EPSILON),
        );

        let lambda = self.wavelength_bounds.sample(self.last_lambda);
        let (camera_ray, _lens_normal, pdf) = camera.sample_we(film_sample, &mut sampler, lambda);
        let _camera_pdf = pdf;

        let mut path: Vec<SurfaceVertex> = vec![SurfaceVertex::new(
            VertexType::Camera,
            camera_ray.time,
            lambda,
            Vec3::ZERO,
            camera_ray.origin,
            camera_ray.direction,
            (0.0, 0.0),
            MaterialId::Camera(0),
            0,
            SingleEnergy::ONE,
            0.0,
            0.0,
            1.0,
        )];

        let _ = random_walk(
            camera_ray,
            lambda,
            self.max_bounces,
            SingleEnergy::ONE,
            TransportMode::Importance,
            sampler,
            &self.world,
            &mut path,
            1,
            &mut profile,
        );

        profile.camera_rays += 1;
        // camera random walk is now stored in path, with length limited to 1 (for now)
        // let vertex_in_scene = path.last().unwrap();
        let radius_squared = 0.005 / (1.0 + sample_id as f32);
        let scaling_factor = PI * radius_squared;

        for vertex_in_scene in path.iter().skip(1) {
            let mut temp_sum = XYZColor::ZERO;
            let mut n = 0;
            // collect photons that are within a certain radius
            for vert in self.photon_map.as_ref().unwrap().photons.iter() {
                let point = vert.point;

                let vec_to_camera: Vec3 = camera_ray.origin - vertex_in_scene.point;
                if vec_to_camera.norm_squared() < 0.001 {
                    continue;
                }
                let distance_squared = (point - vertex_in_scene.point).norm_squared();
                if distance_squared < radius_squared {
                    let wi_global: Vec3 = vert.local_wi;
                    let wo_global: Vec3 = vec_to_camera.normalized();

                    let normal = vert.normal;
                    let frame = TangentFrame::from_normal(normal);

                    let wi = frame.to_local(&wi_global);
                    let wo = frame.to_local(&wo_global);

                    let material = self.world.get_material(vertex_in_scene.material_id);

                    let hit: HitRecord = (*vertex_in_scene).into();
                    let (f, pdf) = material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, wo);

                    if pdf.0 == 0.0 {
                        continue;
                    }

                    temp_sum += XYZColor::from(SingleWavelength::new(
                        lambda,
                        vert.throughput.0 * f / pdf.0
                            * window_function(
                                0.1 * lambda * (1.0 + sample_id as f32),
                                0.1 * vert.lambda * (1.0 + sample_id as f32),
                            )
                            / scaling_factor,
                    ));
                    n += 1;
                }
            }
            if n > 0 {
                sum += temp_sum / n as f32;
            }
        }

        sum
    }
}

// impl SPPMIntegrator {
//     fn color_given_lambda(
//         &self,
//         sampler: &mut Box<dyn Sampler>,
//         camera_sample: ((f32, f32), CameraId),
//         sample_id: usize,
//         mut profile: &mut Profile,
//         lambda: f32,
//     ) -> XYZColor {
//         // naive implementation of SPPM
//         // iterate through all deposited photons and add contributions based on if they are close to the eye vertex in question

//         let camera_id = camera_sample.1;
//         let camera = self.world.get_camera(camera_id as usize);

//         let mut sum = XYZColor::ZERO;
//         // let (direction, camera_pdf) = camera_surface.sample(camera_direction_sample, hit.point);
//         // let direction = direction.normalized();
//         let film_sample = Sample2D::new(
//             (camera_sample.0).0.clamp(0.0, 1.0 - EPSILON),
//             (camera_sample.0).1.clamp(0.0, 1.0 - EPSILON),
//         );

//         let lambda = self.wavelength_bounds.sample(sampler.draw_1d().x);
//         let aperture_sample = sampler.draw_2d();
//         let (camera_ray, _lens_normal, pdf) =
//             camera.sample_we(film_sample, aperture_sample, lambda);
//         let _camera_pdf = pdf;

//         let mut path: Vec<SurfaceVertex> = vec![SurfaceVertex::new(
//             VertexType::Camera,
//             camera_ray.time,
//             lambda,
//             Vec3::ZERO,
//             camera_ray.origin,
//             camera_ray.direction,
//             (0.0, 0.0),
//             MaterialId::Camera(0),
//             0,
//             SingleEnergy::ONE,
//             0.0,
//             0.0,
//             1.0,
//         )];

//         let _ = random_walk(
//             camera_ray,
//             lambda,
//             self.max_bounces,
//             SingleEnergy::ONE,
//             TransportMode::Importance,
//             sampler,
//             &self.world,
//             &mut path,
//             1,
//             &mut profile,
//         );

//         profile.camera_rays += 1;
//         // camera random walk is now stored in path, with length limited to 1 (for now)
//         // let vertex_in_scene = path.last().unwrap();
//         let radius_squared = 0.005 / (1.0 + sample_id as f32);
//         let scaling_factor = PI * radius_squared;

//         for vertex_in_scene in path.iter().skip(1) {
//             let mut temp_sum = XYZColor::ZERO;
//             let mut n = 0;
//             // collect photons that are within a certain radius
//             for vert in self.photon_map.as_ref().unwrap().photons.iter() {
//                 let point = vert.point;

//                 let vec_to_camera: Vec3 = camera_ray.origin - vertex_in_scene.point;
//                 if vec_to_camera.norm_squared() < 0.001 {
//                     continue;
//                 }
//                 let distance_squared = (point - vertex_in_scene.point).norm_squared();
//                 if distance_squared < radius_squared {
//                     let wi_global: Vec3 = vert.local_wi;
//                     let wo_global: Vec3 = vec_to_camera.normalized();

//                     let normal = vert.normal;
//                     let frame = TangentFrame::from_normal(normal);

//                     let wi = frame.to_local(&wi_global);
//                     let wo = frame.to_local(&wo_global);

//                     let material = self.world.get_material(vertex_in_scene.material_id);

//                     let hit: HitRecord = (*vertex_in_scene).into();
//                     let (f, pdf) = material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, wo);

//                     if pdf.0 == 0.0 {
//                         continue;
//                     }

//                     temp_sum += XYZColor::from(SingleWavelength::new(
//                         lambda,
//                         vert.throughput.0 * f / pdf.0
//                             * window_function(
//                                 0.1 * lambda * (1.0 + sample_id as f32),
//                                 0.1 * vert.lambda * (1.0 + sample_id as f32),
//                             )
//                             / scaling_factor,
//                     ));
//                     n += 1;
//                 }
//             }
//             if n > 0 {
//                 sum += temp_sum / n as f32;
//             }
//         }

//         sum
//     }
// }
