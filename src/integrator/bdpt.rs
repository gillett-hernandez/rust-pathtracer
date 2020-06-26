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
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Source {
    Instance,
    Environment,
}
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Type {
    LightSource(Source),
    Light,
    Eye,
    Camera,
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
    pub throughput: SingleEnergy,
    pub pdf_forward: f32,
    pub pdf_backward: f32,
    pub veach_g: f32,
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
        throughput: SingleEnergy,
        pdf_forward: f32,
        pdf_backward: f32,
        veach_g: f32,
    ) -> Self {
        Vertex {
            kind,
            time,
            lambda,
            point,
            normal,
            material_id,
            instance_id,
            throughput,
            pdf_forward,
            pdf_backward,
            veach_g,
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

pub fn veach_g(point0: Point3, cos_i: f32, point1: Point3, cos_o: f32) -> f32 {
    (cos_i * cos_o).abs() / (point1 - point0).norm_squared()
}

pub fn veach_v(world: &Arc<World>, point0: Point3, point1: Point3) -> bool {
    // returns if the points are visible
    let diff = point1 - point0;
    let norm = diff.norm();
    let tmax = norm * 0.95;
    let point0_to_point1 = Ray::new_with_time_and_tmax(point0, diff / norm, 0.0, tmax);
    let hit = world.hit(point0_to_point1, 0.01, tmax);
    // if (point0.x() == 1.0 || point1.x() == 1.0) && !hit.as_ref().is_none() {
    //     // from back wall to something
    //     println!(
    //         "{:?} {:?}, hit was {:?}",
    //         point0,
    //         point1,
    //         hit.as_ref().unwrap()
    //     );
    // }
    hit.is_none()
}

pub fn random_walk(
    mut ray: Ray,
    lambda: f32,
    bounce_limit: u16,
    start_throughput: SingleEnergy,
    trace_type: Type,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<Vertex>,
) {
    let mut beta = start_throughput;
    for vert_idx in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.0, INFINITY) {
            hit.lambda = lambda;
            let frame = TangentFrame::from_normal(hit.normal);
            let wi = frame.to_local(&-ray.direction).normalized();
            let material: &Box<dyn Material> = &world.materials[hit.material as usize];

            // let emission = material.emission(&hit, wi, None);
            // if emission.0 > 0.0 && trace_type == Type::Eye {

            // }
            // wo is generated in tangent space.
            let mut vertex = Vertex::new(
                trace_type,
                hit.time,
                hit.lambda,
                hit.point,
                hit.normal,
                hit.material,
                hit.instance_id,
                beta,
                0.0,
                0.0,
                1.0,
            );
            let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);

            if let Some(wo) = maybe_wo {
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                let pdf = material.value(&hit, wi, wo);
                debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                if pdf.0 < 0.00000001 || pdf.is_nan() {
                    break;
                }

                vertex.pdf_forward = pdf.0 / cos_i;
                if cos_o < 0.00001 {
                    // considered specular
                    vertex.pdf_backward = vertex.pdf_forward;
                } else {
                    vertex.pdf_backward = vertex.pdf_forward / cos_o;
                }

                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);

                vertices.push(vertex);

                let f = material.f(&hit, wi, wo);

                let beta_before_hit = beta;
                beta *= f * cos_i.abs() / pdf.0;

                debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * 0.01 * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                vertex.pdf_forward = 0.0;
                vertex.pdf_backward = 1.0;
                vertex.veach_g = veach_g(hit.point, 1.0, ray.origin, 1.0);
                vertices.push(vertex);
                break;
            }
        }
    }
}

pub fn eval_unweighted_contribution(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s_light_idx: usize,
    eye_path: &Vec<Vertex>,
    t_eye_idx: usize,
) -> (SingleEnergy, f32) {
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
    let g;
    if s_light_idx == 0 {
        // since the eye path actually hit the light in this situation, calculate how much light would be transmitted along that eye path
        let second_to_last_eye_vertex = eye_path[t_eye_idx - 2];
        let last_eye_vertex = eye_path[t_eye_idx - 1];
        let hit_light_material = world.get_material(last_eye_vertex.material_id);

        let normal = last_eye_vertex.normal;
        let frame = TangentFrame::from_normal(normal);
        let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();

        cst = hit_light_material.emission(
            &last_eye_vertex.into(),
            frame.to_local(&wi).normalized(),
            None,
        );
        // if cst.0 > 0.0 {
        //     println!(
        //         "{:?}, {:?}, {:?}",
        //         cst, last_eye_vertex.material_id, last_eye_vertex.instance_id
        //     );
        // }
        g = 1.0;
    } else if t_eye_idx == 0 {
        // since the light path actually directly hit the camera, ignore it for now. maybe add it to a splatting queue once a more sophisticated renderer is implemented
        // but the current renderer can't handle contributions that aren't restricted to the current pixel
        return (SingleEnergy::ZERO, 0.0);
    } else {
        // assume light_path[0] and light_path[1] have had their reflectances fixed to be the light radiance values, as that's what the BDPT algorithm seems to expect.
        // also assume that camera_path[0] throughput is set to the so called We value, which is a measure of the importance of the given camera ray and wavelength sample

        // a valid connection can be made.
        let last_light_vertex = light_path[s_light_idx - 1]; // s_end_v
        let last_eye_vertex = eye_path[t_eye_idx - 1]; // t_end_v

        let light_to_eye_vec = last_eye_vertex.point - last_light_vertex.point;
        let light_to_eye_direction = light_to_eye_vec.normalized();

        let llv_normal = last_light_vertex.normal;
        let llv_frame = TangentFrame::from_normal(llv_normal);
        let llv_world_light_to_eye = light_to_eye_direction;
        let llv_local_light_to_eye = llv_frame.to_local(&llv_world_light_to_eye).normalized();
        let fsl = if s_light_idx == 1 {
            // connected to surface of light
            // issue here. debug.

            let hit_light_material = world.get_material(last_light_vertex.material_id);
            let emission = hit_light_material.emission(
                &last_light_vertex.into(),
                llv_local_light_to_eye,
                None,
            );
            // if last_eye_vertex.instance_id == 8 && emission.0 < 0.1 {
            //     println!(
            //         "{:?} connecting to point {:?} (with wi of {:?})resulted in emission of {}",
            //         last_light_vertex, last_eye_vertex.point, llv_local_light_to_eye, emission.0
            //     );
            // }
            // assert!(emission.0 > 0.0);
            emission
        } else {
            let second_to_last_light_vertex = light_path[s_light_idx - 2];
            let wi = (second_to_last_light_vertex.point - last_light_vertex.point).normalized();
            let hit_material = world.get_material(last_light_vertex.material_id);
            hit_material.f(
                &last_light_vertex.into(),
                llv_frame.to_local(&wi).normalized(),
                llv_local_light_to_eye,
            )
        };

        if fsl == SingleEnergy::ZERO {
            // if last_eye_vertex.instance_id == 8 {
            //     println!("returning 0 at line 281");
            // }
            return (SingleEnergy::ZERO, 0.0);
        }

        let lev_normal = last_eye_vertex.normal;
        let lev_frame = TangentFrame::from_normal(lev_normal);
        let lev_world_eye_to_light = -light_to_eye_direction;
        let lev_local_eye_to_light = lev_frame.to_local(&lev_world_eye_to_light).normalized();
        let fse = if t_eye_idx == 1 {
            // another unsupported situation. see the above note about the current renderer
            // camera.eval_we(last_eye_vertex.point, last_light_vertex.point)
            return (SingleEnergy::ZERO, 0.0);
        } else {
            let second_to_last_eye_vertex = eye_path[t_eye_idx - 2];
            let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
            // let wo = -light_to_eye;
            let hit_material = world.get_material(last_eye_vertex.material_id);
            let reflectance = hit_material.f(
                &last_eye_vertex.into(),
                lev_frame.to_local(&wi).normalized(),
                lev_local_eye_to_light,
            );
            // if last_eye_vertex.instance_id == 8 && reflectance.0 < 0.2 {
            //     println!(
            //         "{:?} connecting to point {:?} (with wi of {:?})resulted in reflectance of {}",
            //         last_light_vertex, last_eye_vertex.point, lev_local_eye_to_light, reflectance.0
            //     );
            // }
            reflectance
        };

        if fse == SingleEnergy::ZERO {
            // if last_eye_vertex.instance_id == 8 {
            //     println!("returning 0 at line 315");
            // }
            return (SingleEnergy::ZERO, 0.0);
        }

        let (cos_i, cos_o) = (
            lev_local_eye_to_light.z().abs(),
            llv_local_light_to_eye.z().abs(),
        );
        g = veach_g(last_eye_vertex.point, cos_i, last_light_vertex.point, cos_o);
        if g == 0.0 {
            // if last_eye_vertex.instance_id == 8 {
            //     println!("returning 0 at line 327");
            // }
            return (SingleEnergy::ZERO, 0.0);
        }

        if !veach_v(world, last_eye_vertex.point, last_light_vertex.point) {
            // not visible
            // if last_eye_vertex.instance_id == 8 {
            //     println!("returning 0 at line 336");
            // }
            return (SingleEnergy::ZERO, 0.0);
        }
        cst = fsl * g * fse;
    }

    (
        last_light_vertex_throughput * cst * last_eye_vertex_throughput,
        g,
    )
}

pub fn eval_mis(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s: usize,
    eye_path: &Vec<Vertex>,
    t: usize,
    veach_g: f32,
    mis_nodes: &Vec<(f32, f32, bool)>,
) -> f32 {
    
    1.0
}

impl Integrator for BDPTIntegrator {
    fn color(&self, sampler: &mut Box<dyn Sampler>, camera_ray: Ray) -> SingleWavelength {
        // setup: decide light, emit ray from light, emit ray from camera, connect light path vertices to camera path vertices.

        let wavelength_sample = sampler.draw_1d();
        let mut light_pick_sample = sampler.draw_1d();

        let scene_light_sampling_probability = self.world.get_env_sampling_probability();

        let sampled;
        let mut light_g_term: f32 = 1.0;

        let start_light_vertex;
        if self.world.lights.len() > 0 && light_pick_sample.x < scene_light_sampling_probability {
            light_pick_sample.x =
                (light_pick_sample.x / scene_light_sampling_probability).clamp(0.0, 1.0);
            let (light, light_pick_pdf) = self.world.pick_random_light(light_pick_sample).unwrap();

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

            let directional_pdf = sampled.2;
            // if delta light, the pdf_forward is only directional_pdf
            let pdf_forward: PDF = directional_pdf / light_g_term;
            let pdf_backward: PDF = light_pick_pdf / light.surface_area(&Transform3::new());

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
        light_path.push(start_light_vertex);

        random_walk(
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
                let (factor, g) =
                    eval_unweighted_contribution(&self.world, &light_path, s, &eye_path, t);
                return SingleWavelength::new(lambda, factor);
            }
        }

        let mis_enabled = false;
        let mut mis_nodes: Vec<(f32, f32, bool)> = Vec::new();
        let mut sum = SingleEnergy::ZERO;
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
                g = new_g;
                let weight = if mis_enabled {
                    eval_mis(&self.world, &light_path, s, &eye_path, t, g, &mis_nodes)
                } else {
                    1.0
                };
                if factor == SingleEnergy::ZERO || weight == 0.0 {
                    continue;
                }
                sum += weight * factor;
            }
        }
        SingleWavelength::new(lambda, sum)
    }
}
