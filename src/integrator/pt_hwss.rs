use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::{
    random_walk, veach_v, HeroEnergy, HeroVertex, LightSourceType, Vertex, VertexType,
};
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};
use crate::math::*;
use crate::world::TransportMode;
// use crate::world::EnvironmentMap;

use crate::integrator::pt::PathTracingIntegrator;
use packed_simd::f32x4;
use std::f32::INFINITY;
use std::sync::Arc;
use utils::random_walk_hero;

pub struct HWSSPathTracingIntegrator {
    pub inner: PathTracingIntegrator,
}

pub fn generate_hero(x: f32, bounds: Bounds1D) -> f32x4 {
    let hero = x * bounds.span();
    let delta = bounds.span() / 4.0;
    let mult = f32x4::new(0.0, 1.0, 2.0, 3.0);
    let wavelengths = bounds.lower + (hero + mult * delta);
    let sub: f32x4 = wavelengths
        .gt(f32x4::splat(bounds.upper))
        .select(f32x4::splat(bounds.span()), f32x4::splat(0.0));
    wavelengths - sub
}

impl SamplerIntegrator for HWSSPathTracingIntegrator {
    fn color(
        &self,
        sampler: &mut Box<dyn Sampler>,
        camera_sample: ((f32, f32), CameraId),
        _sample_id: usize,
        mut profile: &mut Profile,
    ) -> XYZColor {
        profile.camera_rays += 1;

        let lambda = generate_hero(sampler.draw_1d().x, self.inner.wavelength_bounds);
        let mut sum = HeroEnergy(f32x4::splat(0.0));

        let camera_id = camera_sample.1;
        let camera = self.inner.world.get_camera(camera_id as usize);
        let film_sample = Sample2D::new(
            (camera_sample.0).0.clamp(0.0, 1.0 - std::f32::EPSILON),
            (camera_sample.0).1.clamp(0.0, 1.0 - std::f32::EPSILON),
        );
        let aperture_sample = sampler.draw_2d(); // sometimes called aperture sample
        let (camera_ray, _lens_normal, pdf) =
            camera.sample_we(film_sample, aperture_sample, lambda.extract(0));
        let _camera_pdf = pdf;

        let mut path: Vec<HeroVertex> = Vec::with_capacity(1 + self.inner.max_bounces as usize);

        path.push(HeroVertex::new(
            VertexType::Camera,
            camera_ray.time,
            lambda,
            Vec3::ZERO,
            camera_ray.origin,
            camera_ray.direction,
            (0.0, 0.0),
            MaterialId::Camera(0),
            0,
            HeroEnergy(f32x4::splat(1.0)),
            f32x4::splat(0.0),
            f32x4::splat(0.0),
            1.0,
        ));
        let _ = random_walk_hero(
            camera_ray,
            lambda,
            self.inner.max_bounces,
            f32x4::splat(1.0),
            TransportMode::Importance,
            sampler,
            &self.inner.world,
            &mut path,
            self.inner.min_bounces,
            &mut profile,
        );

        for (index, vertex) in path.iter().enumerate() {
            if index == 0 {
                continue;
            }
            let prev_vertex = path[index - 1];
            // for every vertex past the 1st one (which is on the camera), evaluate the direct illumination at that vertex, and if it hits a light evaluate the added energy
            if let VertexType::LightSource(light_source) = vertex.vertex_type {
                if light_source == LightSourceType::Environment {
                    let wo = vertex.local_wi;
                    let uv = direction_to_uv(wo);
                    let mut emission = f32x4::splat(0.0);
                    for i in 0..4 {
                        emission = emission.replace(
                            i,
                            self.inner
                                .world
                                .environment
                                .emission(uv, lambda.extract(i))
                                .0,
                        );
                    }
                    sum.0 += emission * vertex.throughput.0;
                } else {
                    // let hit = HitRecord::from(*vertex);
                    let frame = TangentFrame::from_normal(vertex.normal);
                    let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                    let _maybe_dir_to_next = path
                        .get(index + 1)
                        .map(|v| (v.point - vertex.point).normalized());
                    let wi = frame.to_local(&dir_to_prev);
                    let material = self.inner.world.get_material(vertex.material_id);
                    let mut emission = f32x4::splat(0.0);
                    for i in 0..4 {
                        emission = emission.replace(
                            i,
                            material
                                .emission(lambda.extract(i), vertex.uv, vertex.transport_mode(), wi)
                                .0,
                        );
                    }

                    if emission.gt(f32x4::splat(0.0)).any() {
                        if prev_vertex.pdf_forward.extract(0) <= 0.0
                            || self.inner.light_samples == 0
                        {
                            sum.0 += vertex.throughput.0 * emission;
                            debug_assert!(!sum.0.is_nan().any());
                        } else {
                            let hit_primitive = self.inner.world.get_primitive(vertex.instance_id);
                            // // println!("{:?}", hit);
                            let pdf = f32x4::splat(
                                hit_primitive
                                    .psa_pdf(
                                        prev_vertex.normal
                                            * (vertex.point - prev_vertex.point).normalized(),
                                        prev_vertex.point,
                                        vertex.point,
                                    )
                                    .0,
                            );
                            let weight = power_heuristic_hero(prev_vertex.pdf_forward, pdf);
                            debug_assert!(
                                !pdf.is_nan().any() && !weight.is_nan().any(),
                                "{:?}, {:?}",
                                pdf,
                                weight
                            );
                            sum.0 += vertex.throughput.0 * emission * weight;
                            debug_assert!(!sum.0.is_nan().any());
                        }
                    }
                }
            } else {
                let frame = TangentFrame::from_normal(vertex.normal);
                let dir_to_prev = (prev_vertex.point - vertex.point).normalized();
                let _maybe_dir_to_next = path
                    .get(index + 1)
                    .map(|v| (v.point - vertex.point).normalized());
                let wi = frame.to_local(&dir_to_prev);
                let material = self.inner.world.get_material(vertex.material_id);

                let mut emission = f32x4::splat(0.0);
                for i in 0..4 {
                    emission = emission.replace(
                        i,
                        material
                            .emission(lambda.extract(i), vertex.uv, vertex.transport_mode(), wi)
                            .0,
                    );
                }

                if emission.gt(f32x4::splat(0.0)).any() {
                    // this will likely never get triggered, since hitting a light source is handled in the above branch
                    if prev_vertex.pdf_forward.extract(0) <= 0.0 || self.inner.light_samples == 0 {
                        sum.0 += vertex.throughput.0 * emission;
                        debug_assert!(!sum.0.is_nan().any());
                    } else {
                        let hit_primitive = self.inner.world.get_primitive(vertex.instance_id);
                        // // println!("{:?}", hit);
                        let pdf = f32x4::splat(
                            hit_primitive
                                .psa_pdf(
                                    prev_vertex.normal
                                        * (vertex.point - prev_vertex.point).normalized(),
                                    prev_vertex.point,
                                    vertex.point,
                                )
                                .0,
                        );
                        let weight = power_heuristic_hero(prev_vertex.pdf_forward, pdf);
                        debug_assert!(
                            !pdf.is_nan().any() && !weight.is_nan().any(),
                            "{:?}, {:?}",
                            pdf,
                            weight
                        );
                        sum.0 += vertex.throughput.0 * emission * weight;
                        debug_assert!(!sum.0.is_nan().any());
                    }
                }

                if self.inner.light_samples > 0 {
                    let mut light_contribution = f32x4::splat(0.0);
                    for i in 0..4 {
                        light_contribution = light_contribution.replace(
                            i,
                            self.inner
                                .estimate_direct_illumination_with_loop(
                                    lambda.extract(i),
                                    &vertex.into_hit_w_lane(i),
                                    &frame,
                                    wi,
                                    material,
                                    SingleEnergy(vertex.throughput.0.extract(i)),
                                    sampler,
                                    &mut profile,
                                )
                                .0,
                        );
                    }
                    // println!("light contribution: {:?}", light_contribution);
                    sum.0 += light_contribution / (self.inner.light_samples as f32);
                    debug_assert!(
                        !sum.0.is_nan().any(),
                        "{:?} {:?}",
                        light_contribution,
                        self.inner.light_samples
                    );
                }
            }
            if self.inner.only_direct {
                break;
            }
        }

        let mut c = XYZColor::from(SingleWavelength::new(
            lambda.extract(0),
            SingleEnergy(sum.0.extract(0) / 4.0),
        ));
        c += XYZColor::from(SingleWavelength::new(
            lambda.extract(1),
            SingleEnergy(sum.0.extract(1) / 4.0),
        ));
        c += XYZColor::from(SingleWavelength::new(
            lambda.extract(2),
            SingleEnergy(sum.0.extract(2) / 4.0),
        ));
        c += XYZColor::from(SingleWavelength::new(
            lambda.extract(3),
            SingleEnergy(sum.0.extract(3) / 4.0),
        ));
        c
    }
}
