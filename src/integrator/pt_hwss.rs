use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::integrator::utils::{
    random_walk, veach_v, HeroEnergy, HeroSurfaceVertex, LightSourceType, SurfaceVertex, VertexType,
};
use crate::integrator::*;
use crate::materials::{Material, MaterialEnum, MaterialId};
use crate::math::*;
use crate::mediums::{Medium, MediumEnum};
use crate::world::TransportMode;
// use crate::world::EnvironmentMap;

use crate::integrator::pt::PathTracingIntegrator;
use packed_simd::f32x4;
use std::f32::INFINITY;
use std::sync::Arc;
use utils::{random_walk_hero, random_walk_medium_hero, HeroMediumVertex, HeroVertex};

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

pub fn medium_direct_illumination(
    world: &Arc<World>,
    sample_world: bool,
    medium: &MediumEnum,
    lambda: f32x4,
    vertex: &HeroMediumVertex,
    mediums: &[usize],
    light_pick_sample: Sample1D,
    additional_light_sample: Sample2D,
    profile: &mut Profile,
) -> HeroEnergy {
    if sample_world {
        // light_contribution += direct_illumination_from_world;
        let (uv, light_pdf) = world
            .environment
            .sample_env_uv_given_wavelength(additional_light_sample, lambda.extract(0));
        // direction is the direction to the sampled point on the environment
        let wo = uv_to_direction(uv);

        profile.shadow_rays += 1;
        let mut point = vertex.point;
        loop {
            if let Some(mut _light_hit) = world.hit(Ray::new(point, wo), 0.00001, INFINITY) {
                // handle case where we intended to hit the world but instead hit a light?
                match world.get_material(_light_hit.material) {
                    MaterialEnum::PassthroughFilter(_) => {
                        // need to update mediums for this step, and update transmittance.
                        point = _light_hit.point + 0.00001 * wo;
                        continue;
                    }
                    _ => {
                        return HeroEnergy::ZERO;
                    }
                }
            }
            break;
        }
        let mut contribution = f32x4::splat(0.0);
        for i in 0..4 {
            let f_and_pdf = medium.p(lambda.extract(i), vertex.uvw, vertex.wi, wo);

            // successfully hit nothing, which is to say, hit the world
            let emission = world.environment.emission(uv, lambda.extract(i));

            let weight = power_heuristic(light_pdf.0, f_and_pdf);
            contribution = contribution.replace(
                i,
                f_and_pdf * vertex.throughput.0.extract(i) * emission.0 * weight / light_pdf.0,
            );

            debug_assert!(
                contribution.is_finite().all(),
                "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
                i,
                f_and_pdf,
                vertex.throughput,
                emission,
                weight,
                light_pdf
            );
        }
        debug_assert!(contribution.is_finite().all(), "{:?}", contribution);
        HeroEnergy(contribution)
    } else {
        // light_contribution += direct_illumination_from_light;
        if let Some((light, light_pick_pdf)) = world.pick_random_light(light_pick_sample) {
            // determine pick pdf
            // as of now the pick pdf is just num lights, however if it were to change this would be where it should change.
            // sample the primitive from hit_point
            // let (direction, light_pdf) = light.sample(additional_light_sample, hit.point);
            let (point_on_light, normal, light_area_pdf) =
                light.sample_surface(additional_light_sample);
            debug_assert!(light_area_pdf.0.is_finite());
            if light_area_pdf.0 == 0.0 {
                return HeroEnergy::ZERO;
            }
            // direction is from shading point to light
            let wo = (point_on_light - vertex.point).normalized();
            // direction is already in world space.
            // direction is also oriented away from the shading point already, so no need to negate directions until later.

            let light_vertex_wi = TangentFrame::from_normal(normal).to_local(&(-wo));

            let dropoff = light_vertex_wi.z().abs();
            if dropoff == 0.0 {
                return HeroEnergy::ZERO;
            }

            let pdf = light.psa_pdf(1.0, vertex.point, point_on_light);
            let light_pdf = pdf * light_pick_pdf; // / light_vertex_wi.z().abs();
            if light_pdf.0 == 0.0 {
                // println!("light pdf was 0");
                // go to next pick
                return HeroEnergy::ZERO;
            }
            let light_material = world.get_material(light.get_material_id());
            let mut contribution = f32x4::splat(0.0);
            for i in 0..4 {
                let f_and_pdf = medium.p(lambda.extract(i), vertex.uvw, vertex.wi, wo);
                let emission = light_material.emission(
                    lambda.extract(i),
                    (0.0, 0.0),
                    vertex.transport_mode(),
                    light_vertex_wi,
                );
                // this should be the same as the other method, but maybe not.
                if emission.0 == 0.0 {
                    //
                } else {
                    profile.shadow_rays += 1;
                    let mut point = vertex.point;
                    let direction = (point_on_light - vertex.point).normalized();
                    loop {
                        if let Some(light_hit) =
                            world.hit(Ray::new(point, direction), 0.00001, INFINITY)
                        {
                            match world.get_material(light_hit.material) {
                                MaterialEnum::PassthroughFilter(_) => {
                                    // need to update mediums for this step, and update transmittance.
                                    point = light_hit.point + 0.00001 * direction;
                                    continue;
                                }
                                _ => {
                                    return HeroEnergy::ZERO;
                                }
                            }
                        }
                        break;
                    }
                    let weight = power_heuristic(light_pdf.0, f_and_pdf);

                    debug_assert!(emission.0 >= 0.0);
                    // successful_light_samples += 1;
                    contribution = contribution.replace(
                        i,
                        f_and_pdf * vertex.throughput.0.extract(i) * dropoff * emission.0 * weight
                            / light_pdf.0,
                    );
                    debug_assert!(
                        contribution.is_finite().all(),
                        "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
                        i,
                        f_and_pdf,
                        vertex.throughput,
                        emission,
                        weight,
                        light_pdf
                    );
                }
            }
            debug_assert!(contribution.is_finite().all(), "{:?}", contribution);
            HeroEnergy(contribution)
        } else {
            HeroEnergy::ZERO
        }
    }
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

        path.push(HeroVertex::Surface(HeroSurfaceVertex::new(
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
        )));
        let _ = random_walk_medium_hero(
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
            match vertex {
                HeroVertex::Surface(vertex) => {
                    if let VertexType::LightSource(light_source) = vertex.vertex_type {
                        // if light source hit
                        if light_source == LightSourceType::Environment {
                            // if light source "surface" hit but also environment
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
                            debug_assert!(
                                sum.0.is_finite().all(),
                                "{:?} {:?}",
                                vertex.throughput.0,
                                emission,
                            );
                        } else {
                            // if light source surface hit but not environment
                            // let hit = HitRecord::from(*vertex);
                            let frame = TangentFrame::from_normal(vertex.normal);
                            let dir_to_prev = (prev_vertex.point() - vertex.point).normalized();
                            let _maybe_dir_to_next = path
                                .get(index + 1)
                                .map(|v| (v.point() - vertex.point).normalized());
                            let wi = frame.to_local(&dir_to_prev);
                            let material = self.inner.world.get_material(vertex.material_id);
                            let mut emission = f32x4::splat(0.0);
                            for i in 0..4 {
                                emission = emission.replace(
                                    i,
                                    material
                                        .emission(
                                            lambda.extract(i),
                                            vertex.uv,
                                            vertex.transport_mode(),
                                            wi,
                                        )
                                        .0,
                                );
                            }

                            if emission.gt(f32x4::splat(0.0)).any() {
                                if prev_vertex.pdf_forward().extract(0) <= 0.0
                                    || self.inner.light_samples == 0
                                {
                                    sum.0 += vertex.throughput.0 * emission;
                                    debug_assert!(!sum.0.is_nan().any());
                                    debug_assert!(
                                        sum.0.is_finite().all(),
                                        "{:?} {:?}",
                                        vertex.throughput.0,
                                        emission,
                                    );
                                } else {
                                    let hit_primitive =
                                        self.inner.world.get_primitive(vertex.instance_id);
                                    // // println!("{:?}", hit);
                                    let pdf = f32x4::splat(
                                        hit_primitive
                                            .psa_pdf(
                                                prev_vertex.cos(
                                                    (vertex.point - prev_vertex.point())
                                                        .normalized(),
                                                ),
                                                prev_vertex.point(),
                                                vertex.point,
                                            )
                                            .0,
                                    );
                                    let weight =
                                        power_heuristic_hero(prev_vertex.pdf_forward(), pdf);
                                    debug_assert!(
                                        !pdf.is_nan().any() && !weight.is_nan().any(),
                                        "{:?}, {:?}",
                                        pdf,
                                        weight
                                    );
                                    sum.0 += vertex.throughput.0 * emission * weight;
                                    debug_assert!(!sum.0.is_nan().any());
                                    debug_assert!(
                                        sum.0.is_finite().all(),
                                        "{:?} {:?} {:?}",
                                        vertex.throughput.0,
                                        emission,
                                        weight
                                    );
                                }
                            }
                        }
                    } else {
                        // non light source surface hit
                        let frame = TangentFrame::from_normal(vertex.normal);
                        let dir_to_prev = (prev_vertex.point() - vertex.point).normalized();
                        let _maybe_dir_to_next = path
                            .get(index + 1)
                            .map(|v| (v.point() - vertex.point).normalized());
                        let wi = frame.to_local(&dir_to_prev);
                        let material = self.inner.world.get_material(vertex.material_id);

                        let mut emission = f32x4::splat(0.0);
                        for i in 0..4 {
                            emission = emission.replace(
                                i,
                                material
                                    .emission(
                                        lambda.extract(i),
                                        vertex.uv,
                                        vertex.transport_mode(),
                                        wi,
                                    )
                                    .0,
                            );
                        }

                        if emission.gt(f32x4::splat(0.0)).any() {
                            // this will likely never get triggered, since hitting a light source is handled in the above branch
                            if prev_vertex.pdf_forward().extract(0) <= 0.0
                                || self.inner.light_samples == 0
                            {
                                sum.0 += vertex.throughput.0 * emission;
                                debug_assert!(!sum.0.is_nan().any());
                                debug_assert!(
                                    sum.0.is_finite().all(),
                                    "{:?} {:?}",
                                    vertex.throughput.0,
                                    emission,
                                );
                            } else {
                                let hit_primitive =
                                    self.inner.world.get_primitive(vertex.instance_id);
                                // // println!("{:?}", hit);
                                let pdf = f32x4::splat(
                                    hit_primitive
                                        .psa_pdf(
                                            prev_vertex.cos(
                                                (vertex.point - prev_vertex.point()).normalized(),
                                            ),
                                            prev_vertex.point(),
                                            vertex.point,
                                        )
                                        .0,
                                );
                                let weight = power_heuristic_hero(prev_vertex.pdf_forward(), pdf);
                                debug_assert!(
                                    !pdf.is_nan().any() && !weight.is_nan().any(),
                                    "{:?}, {:?}",
                                    pdf,
                                    weight
                                );
                                sum.0 += vertex.throughput.0 * emission * weight;
                                debug_assert!(!sum.0.is_nan().any());
                                debug_assert!(
                                    sum.0.is_finite().all(),
                                    "{:?} {:?} {:?}",
                                    vertex.throughput.0,
                                    emission,
                                    weight
                                );
                            }
                        }

                        if self.inner.light_samples > 0
                            && material.get_name() != "PassthroughFilter"
                        {
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
                                sum.0.is_finite().all(),
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
                HeroVertex::Medium(vertex) => {
                    match vertex.vertex_type {
                        VertexType::LightSource(LightSourceType::Environment) => {
                            // all light source env hits are through surface vertices, not medium vertices
                            panic!();
                        }
                        VertexType::LightSource(_) => {
                            // emissive volume? skip for now
                            break;
                        }
                        VertexType::Eye | VertexType::Light => {
                            // main case
                            // evaluate direct illumination to the medium point.
                            // let medium = &self.inner.world.mediums[vertex.medium_id - 1];

                            // let mut light_contribution = HeroEnergy(f32x4::splat(0.0));
                            // let env_sampling_probability =
                            //     self.inner.world.get_env_sampling_probability();
                            // if self.inner.world.lights.len() == 0 && env_sampling_probability == 0.0
                            // {
                            //     // do nothing. direct illumination is 0 due to env sampling probability being 0
                            // } else {
                            //     for _i in 0..self.inner.light_samples {
                            //         let (light_pick_sample, sample_world) = sampler
                            //             .draw_1d()
                            //             .choose(env_sampling_probability, true, false);
                            //         // decide whether to sample the lights or the world
                            //         let contribution = medium_direct_illumination(
                            //             &self.inner.world,
                            //             sample_world,
                            //             medium,
                            //             lambda,
                            //             vertex,
                            //             light_pick_sample,
                            //             sampler.draw_2d(),
                            //             &mut profile,
                            //         );
                            //         light_contribution += contribution;
                            //     }
                            // }
                            // sum +=
                            //     HeroEnergy(light_contribution.0 / self.inner.light_samples as f32);
                            // debug_assert!(
                            //     sum.0.is_finite().all(),
                            //     "{:?} {:?}",
                            //     light_contribution,
                            //     self.inner.light_samples
                            // );
                        }
                        VertexType::Camera => {
                            // hit camera. since this is path tracing, we don't care. in fact, since the pt integrator version of the world doesn't even have a camera "surface" in it, this is impossible
                            panic!();
                        }
                    }
                }
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
        debug_assert!(c.0.is_finite().all(), "{:?}", sum);
        c
    }
}
