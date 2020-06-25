use crate::world::World;
// use crate::config::Settings;
use crate::aabb::HasBoundingBox;
use crate::hittable::{HitRecord, Hittable};
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
    pub time: f32,
    pub lambda: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: MaterialId,
    pub instance_id: usize,
    pub pdf: PDF,
    pub reflectance: f32,
    pub throughput: SingleEnergy,
}

impl Vertex {
    pub fn new(
        kind: Type,
        time: f32,
        lambda: f32,
        point: Point3,
        normal: Vec3,
        material_id: MaterialId,
        instance_id: usize,
        pdf: PDF,
        reflectance: f32,
        throughput: SingleEnergy,
    ) -> Self {
        Vertex {
            kind,
            time,
            lambda,
            point,
            normal,
            material_id,
            instance_id,
            pdf,
            reflectance,
            throughput,
        }
    }
}

impl From<Vertex> for HitRecord {
    fn from(data: Vertex) -> Self {
        HitRecord::new(
            data.time,
            data.point,
            (0.0, 0.0),
            data.lambda,
            data.normal,
            data.material_id,
            data.instance_id,
        )
    }
}

pub struct BDPTIntegrator {
    pub max_bounces: u16,
    pub world: Arc<World>,
    pub specific_pair: Option<(usize, usize)>,
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
            let id = hit.material as usize;
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

                vertices[i as usize] = Vertex::new(
                    trace_type,
                    hit.time,
                    hit.lambda,
                    hit.point,
                    hit.normal,
                    id as MaterialId,
                    hit.instance_id,
                    pdf,
                    f.0,
                    beta,
                );

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

pub fn veach_g(eye_point: Point3, cos_i: f32, light_point: Point3, cos_o: f32) -> f32 {
    ((eye_point - light_point).norm_squared() * cos_i * cos_o).abs()
}

pub fn veach_v(world: &Arc<World>, eye_point: Point3, light_point: Point3) -> bool {
    let diff = light_point - eye_point;
    let norm = diff.norm();
    let cam_to_light = Ray::new(eye_point, diff / norm);

    let tmax = norm * 0.99;
    world.hit(cam_to_light, 0.0, tmax).is_none()
}

pub fn eval_unweighted_contribution(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s_light_idx: usize,
    eye_path: &Vec<Vertex>,
    t_eye_idx: usize,
) -> SingleEnergy {
    let last_light_vertex_throughput = if s_light_idx == 0 {
        SingleEnergy::ONE
    } else {
        light_path[s_light_idx - 1].throughput
    };

    let last_eye_vertex_throughput = if t_eye_idx == 0 {
        SingleEnergy::ONE
    } else {
        eye_path[t_eye_idx - 1].throughput
    };

    let cst: SingleEnergy;
    if s_light_idx == 0 {
        // since the eye path actually hit the light in this situation, calculate how much light would be transmitted along that eye path
        let second_to_last_eye_vertex = eye_path[t_eye_idx - 2];
        let last_eye_vertex = eye_path[t_eye_idx - 1];
        let hit_light_material = world.get_material(last_eye_vertex.material_id);

        let normal = last_eye_vertex.normal;
        let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();

        cst = hit_light_material.emission(&last_eye_vertex.into(), wi, None);
    } else if t_eye_idx == 0 {
        // since the light path actually directly hit the camera, ignore it for now. maybe add it to a splatting queue once a more sophisticated renderer is implemented
        // but the current renderer can't handle contributions that aren't restricted to the current pixel
        return SingleEnergy::ZERO;
    } else {
        // assume light_path[0] and light_path[1] have had their reflectances fixed to be the light radiance values, as that's what the BDPT algorithm seems to expect.
        // also assume that camera_path[0] throughput is set to the so called We value, which is a measure of the importance of the given camera ray and wavelength sample

        // a valid connection can be made.
        let last_light_vertex = light_path[s_light_idx - 1]; // s_end_v
        let last_eye_vertex = eye_path[t_eye_idx - 1]; // t_end_v

        let light_to_eye_vec = last_eye_vertex.point - last_light_vertex.point;
        let light_to_eye_direction = light_to_eye_vec.normalized();

        let fsl = if s_light_idx == 1 {
            let hit_light_material = world.get_material(last_light_vertex.material_id);

            let normal = last_light_vertex.normal;
            let wi = light_to_eye_direction;

            hit_light_material.emission(&last_light_vertex.into(), wi, None)
        } else {
            let second_to_last_light_vertex = light_path[s_light_idx - 2];
            let wi = light_to_eye_direction;
            let wo = (second_to_last_light_vertex.point - last_light_vertex.point).normalized();
            let hit_material = world.get_material(last_light_vertex.material_id);
            hit_material.f(&last_light_vertex.into(), wi, wo)
        };

        if fsl == SingleEnergy::ZERO {
            return SingleEnergy::ZERO;
        }

        let fse = if t_eye_idx == 1 {
            // another unsupported situation. see the above note about the current renderer
            // camera.eval_we(last_eye_vertex.point, last_light_vertex.point)
            return SingleEnergy::ZERO;
        } else {
            let second_to_last_eye_vertex = eye_path[t_eye_idx - 2];
            let wi = -light_to_eye_direction;
            let wo = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
            let hit_material = world.get_material(last_eye_vertex.material_id);
            hit_material.f(&last_eye_vertex.into(), wi, wo)
        };

        if fse == SingleEnergy::ZERO {
            return SingleEnergy::ZERO;
        }

        let (cos_i, cos_o) = (
            (light_to_eye_direction * last_eye_vertex.normal).abs(),
            (light_to_eye_direction * last_light_vertex.normal).abs(),
        );
        let g = veach_g(last_eye_vertex.point, cos_i, last_light_vertex.point, cos_o);
        if g == 0.0 {
            return SingleEnergy::ZERO;
        }

        if !veach_v(world, last_eye_vertex.point, last_light_vertex.point) {
            // not visible
            return SingleEnergy::ZERO;
        }
        cst = fsl * g * fse;
    }
    last_light_vertex_throughput * cst * last_eye_vertex_throughput
}

pub fn eval_mis(world: &Arc<World>, light_path: &Vec<Vertex>, eye_path: &Vec<Vertex>) -> Vec<f32> {
    Vec::new()
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

        // idea: do limited branching and store vertices in a tree format that easily allows for traversal and connections
        let mut light_path: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize - 1);
        let mut eye_path: Vec<Vertex> = Vec::with_capacity(self.max_bounces as usize - 1);

        random_walk(
            camera_ray,
            lambda,
            self.max_bounces - 1,
            Type::Eye,
            sampler,
            &self.world,
            &mut eye_path,
        );
        random_walk(
            light_ray,
            lambda,
            self.max_bounces - 1,
            Type::Light,
            sampler,
            &self.world,
            &mut light_path,
        );

        SingleWavelength::BLACK
    }
}
