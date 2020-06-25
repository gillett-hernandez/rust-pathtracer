use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;
use crate::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;

// use std::f32::INFINITY;
use std::sync::Arc;

use crate::integrator::Integrator;
#[derive(Debug, Copy, Clone)]
pub enum Type {
    Light,
    Eye,
}

#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pub kind: Type,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: MaterialId,
    pub pdf: PDF,
}

impl Vertex {
    pub fn new(kind: Type, point: Point3, normal: Vec3, material_id: MaterialId, pdf: PDF) -> Self {
        Vertex {
            kind,
            point,
            normal,
            material_id,
            pdf,
        }
    }
}

pub struct BDPTIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
}

pub fn random_walk(
    mut ray: Ray,
    lambda: f32,
    bounce_limit: u16,
    trace_type: Type,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<Vertex>,
) {
    // match trace_type {
    //     Kind::Light => {

    //         }
    //     }
    //     Kind::Eye => {}
    // }
    let mut beta = SingleEnergy::ONE;
    for i in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.0, INFINITY) {
            hit.lambda = lambda;
            let id = match hit.material {
                Some(id) => id as usize,
                None => 0,
            };
            let frame = TangentFrame::from_normal(hit.normal);
            let wi = frame.to_local(&-ray.direction).normalized();
            let material: &Box<dyn Material> = &world.materials[id as usize];

            // // wo is generated in tangent space.
            let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);

            if let Some(wo) = maybe_wo {
                let pdf = material.value(&hit, wi, wo);
                debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                if pdf.0 < 0.00000001 || pdf.is_nan() {
                    break;
                }
                let cos_i = wo.z();

                let f = material.f(&hit, wi, wo);
                beta *= f * cos_i.abs() / pdf.0;
                debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);

                vertices[i as usize] =
                    Vertex::new(trace_type, hit.point, hit.normal, id as MaterialId, pdf);

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray

                ray = Ray::new(
                    hit.point + hit.normal * 0.001 * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                break;
            }
        }
    }
}

pub fn veach_g(camera_point: Point3, cos_i: f32, light_point: Point3, cos_o: f32) -> f32 {
    ((camera_point - light_point).norm_squared() * cos_i * cos_o).abs()
}

pub fn veach_v(world: &Arc<World>, camera_point: Point3, light_point: Point3) -> bool {
    let diff = (light_point - camera_point);
    let norm = diff.norm();
    let cam_to_light = Ray::new(camera_point, diff / norm);

    let tmax = norm * 0.99;
    world.hit(cam_to_light, 0.0, tmax).is_none()
}

impl Integrator for BDPTIntegrator {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength {
        // setup: decide light, emit ray from light, emit ray from camera, connect light path vertices to camera path vertices.

        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();

        let scene_light_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;
        let mut light_g_term: f32 = 1.0;

        if self.world.lights.len() > 0 && light_pick_sample.x < scene_light_sampling_probability {
            light_pick_sample.x =
                (light_pick_sample.x / scene_light_sampling_probability).clamp(0.0, 1.0);
            let light = self.world.pick_random_light(light_pick_sample).unwrap();

            // if we picked a light
            let (light_surface_point, light_surface_normal) =
                light.sample_surface(sampler.draw_2d());

            let mat_id = match light.get_material_id() {
                Some(id) => id as usize,
                None => 0,
            };
            let material: &Box<dyn Material> = &self.world.materials[mat_id as usize];
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
            light_g_term = (light_surface_normal * (&sampled.0).direction).abs();
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
        };

        let light_ray = sampled.0;
        let lambda = sampled.1.lambda;
        let radiance = sampled.1.energy;

        let mut light_vertices: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize - 1);
        let mut eye_vertices: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize - 1);

        random_walk(
            camera_ray,
            lambda,
            self.max_bounces - 1,
            Type::Eye,
            sampler,
            &self.world,
            &mut eye_vertices,
        );
        random_walk(
            light_ray,
            lambda,
            self.max_bounces - 1,
            Type::Light,
            sampler,
            &self.world,
            &mut light_vertices,
        );

        SingleWavelength::BLACK
    }
}
