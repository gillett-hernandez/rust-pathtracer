use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;

use std::sync::Arc;

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
    for _ in 0..bounce_limit {
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
                // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                let pdf = material.value(&hit, wi, wo);
                debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                if pdf.0 < 0.00000001 || pdf.is_nan() {
                    break;
                }

                vertex.pdf_forward = pdf.0 / cos_i;

                if cos_i < 0.00001 {
                    // considered specular
                    vertex.pdf_backward = vertex.pdf_forward;
                } else {
                    // eval pdf in reverse direction
                    vertex.pdf_backward = material.value(&hit, wo, wi).0 / cos_o;
                }

                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);

                vertices.push(vertex);

                let f = material.f(&hit, wi, wo);

                // let beta_before_hit = beta;
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
    s: usize,
    eye_path: &Vec<Vertex>,
    t: usize,
) -> (SingleEnergy, f32) {
    let last_light_vertex_throughput = if s == 0 {
        SingleEnergy::ONE
    } else {
        light_path[s - 1].throughput
    };

    let last_eye_vertex_throughput = if t == 0 {
        SingleEnergy::ONE
    } else {
        eye_path[t - 1].throughput
    };

    let cst: SingleEnergy;
    let g;
    if s == 0 {
        // since the eye path actually hit the light in this situation, calculate how much light would be transmitted along that eye path
        let second_to_last_eye_vertex = eye_path[t - 2];
        let last_eye_vertex = eye_path[t - 1];
        let hit_light_material = world.get_material(last_eye_vertex.material_id);

        let normal = last_eye_vertex.normal;
        let frame = TangentFrame::from_normal(normal);
        let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();

        cst = hit_light_material.emission(
            &last_eye_vertex.into(),
            frame.to_local(&wi).normalized(),
            None,
        );
        g = 1.0;
    } else if t == 0 {
        // since the light path actually directly hit the camera, ignore it for now. maybe add it to a splatting queue once a more sophisticated renderer is implemented
        // but the current renderer can't handle contributions that aren't restricted to the current pixel
        return (SingleEnergy::ZERO, 0.0);
    } else {
        // assume light_path[0] and light_path[1] have had their reflectances fixed to be the light radiance values, as that's what the BDPT algorithm seems to expect.
        // also assume that camera_path[0] throughput is set to the so called We value, which is a measure of the importance of the given camera ray and wavelength sample

        // a valid connection can be made.
        let last_light_vertex = light_path[s - 1]; // s_end_v
        let last_eye_vertex = eye_path[t - 1]; // t_end_v

        let light_to_eye_vec = last_eye_vertex.point - last_light_vertex.point;
        let light_to_eye_direction = light_to_eye_vec.normalized();

        // llv means Last Light Vertex
        let llv_normal = last_light_vertex.normal;
        let llv_frame = TangentFrame::from_normal(llv_normal);
        let llv_world_light_to_eye = light_to_eye_direction;
        let llv_local_light_to_eye = llv_frame.to_local(&llv_world_light_to_eye).normalized();
        let fsl = if s == 1 {
            // connected to surface of light

            let hit_light_material = world.get_material(last_light_vertex.material_id);
            let emission = hit_light_material.emission(
                &last_light_vertex.into(),
                llv_local_light_to_eye,
                None,
            );
            emission
        } else {
            let second_to_last_light_vertex = light_path[s - 2];
            let wi = (second_to_last_light_vertex.point - last_light_vertex.point).normalized();
            let hit_material = world.get_material(last_light_vertex.material_id);
            hit_material.f(
                &last_light_vertex.into(),
                llv_frame.to_local(&wi).normalized(),
                llv_local_light_to_eye,
            )
        };

        if fsl == SingleEnergy::ZERO {
            return (SingleEnergy::ZERO, 0.0);
        }

        // lev means Last Eye Vertex
        let lev_normal = last_eye_vertex.normal;
        let lev_frame = TangentFrame::from_normal(lev_normal);
        let lev_world_eye_to_light = -light_to_eye_direction;
        let lev_local_eye_to_light = lev_frame.to_local(&lev_world_eye_to_light).normalized();
        let fse = if t == 1 {
            // another unsupported situation. see the above note about the current renderer
            // camera.eval_we(last_eye_vertex.point, last_light_vertex.point)
            return (SingleEnergy::ZERO, 0.0);
        } else {
            let second_to_last_eye_vertex = eye_path[t - 2];
            let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
            // let wo = -light_to_eye;
            let hit_material = world.get_material(last_eye_vertex.material_id);
            let reflectance = hit_material.f(
                &last_eye_vertex.into(),
                lev_frame.to_local(&wi).normalized(),
                lev_local_eye_to_light,
            );
            reflectance
        };

        if fse == SingleEnergy::ZERO {
            return (SingleEnergy::ZERO, 0.0);
        }

        let (cos_i, cos_o) = (
            lev_local_eye_to_light.z().abs(), // these are cosines relative to their surface normals btw.
            llv_local_light_to_eye.z().abs(), // i.e. eye_to_light.dot(eye_vertex_normal) and light_to_eye.dot(light_vertex_normal)
        );
        g = veach_g(last_eye_vertex.point, cos_i, last_light_vertex.point, cos_o);
        if g == 0.0 {
            return (SingleEnergy::ZERO, 0.0);
        }

        if !veach_v(world, last_eye_vertex.point, last_light_vertex.point) {
            // not visible
            return (SingleEnergy::ZERO, 0.0);
        }
        cst = fsl * g * fse;
    }

    (
        last_light_vertex_throughput * cst * last_eye_vertex_throughput,
        g,
    )
}

pub struct MISNode {
    pdf_toward_light: f32,
    pdf_toward_eye: f32,
    specular: bool,
}

impl MISNode {
    pub fn new(ptl: f32, pte: f32, specular: bool) -> Self {
        MISNode {
            pdf_toward_light: ptl,
            pdf_toward_eye: pte,
            specular,
        }
    }
}

pub fn eval_mis(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s: usize,
    eye_path: &Vec<Vertex>,
    t: usize,
    veach_g: f32,
    mis_nodes: &mut Vec<MISNode>,
) -> f32 {
    let mut pdf_s_end_forward = 0.0;
    let mut pdf_s_end_backward = 0.0;
    let mut pdf_t_end_forward = 0.0;
    let mut pdf_t_end_backward = 0.0;
    if s == 0 {
        // eye path end vertex is light.
        // this is the same as a direct illumination case or the same as the eye path accidentally hitting the light in normal pt
        let last_eye_vertex = eye_path[t - 1];
        let point = last_eye_vertex.point;
        let normal = last_eye_vertex.normal;
        let light = world.get_primitive(last_eye_vertex.instance_id);
        let _light_material = world.get_material(last_eye_vertex.material_id);
        let pick_pdf = 1.0 / (world.lights.len() as f32);
        // pdf_t_end_backward
        let pdf_position = 1.0 / light.surface_area(&Transform3::new());
        pdf_t_end_forward = pick_pdf * pdf_position;
        // let _frame = TangentFrame::from_normal(normal);
        let second_to_last_eye_vertex = eye_path[t - 2];
        pdf_t_end_backward = light.pdf(normal, point, second_to_last_eye_vertex.point).0
            / (normal * (second_to_last_eye_vertex.point - point).normalized());
    } else if t == 0 {
        // light path end vertex is camera lens.
        // not supported
        pdf_s_end_forward = 1.0;
        pdf_s_end_backward = 1.0;
    } else {
        // s and t are both greater than 0
        let last_light_vertex = light_path[s - 1]; // s_end
        let last_eye_vertex = eye_path[t - 1]; // t_end
        let llv_normal = last_light_vertex.normal;
        let llv_frame = TangentFrame::from_normal(llv_normal);
        let lev_normal = last_eye_vertex.normal;
        let lev_frame = TangentFrame::from_normal(lev_normal);

        let light_to_eye = last_eye_vertex.point - last_light_vertex.point;
        // let eye_to_light = (last_light_vertex.point - last_eye_vertex.point);
        // let eye_to_light = -light_to_eye;
        let light_to_eye_direction = light_to_eye.normalized();
        let eye_to_light_direction = -light_to_eye_direction;

        let llv_wo_local = llv_frame.to_local(&light_to_eye_direction).normalized();
        let lev_wo_local = lev_frame.to_local(&eye_to_light_direction).normalized();

        if s == 1 {
            // eye path connecting to specific point on lights' surface
            // evaluate pdf of sampling specific direction from specific point on lights surface
            // let mat_id = last_light_vertex.material_id;
            let light_prim = world.get_primitive(last_light_vertex.instance_id);
            // let light_mat = world.get_material(mat_id);

            let pdf_w = light_prim
                .pdf(llv_normal, last_eye_vertex.point, last_light_vertex.point)
                .0;
            pdf_s_end_forward = pdf_w / (llv_normal * light_to_eye_direction);
            pdf_s_end_backward = last_light_vertex.pdf_backward;
        } else {
            let second_to_last_light_vertex = light_path[s - 2];
            let last_light_to_prev =
                (second_to_last_light_vertex.point - last_light_vertex.point).normalized();
            let direction = llv_frame.to_local(&last_light_to_prev);
            let as_hitrecord = HitRecord::from(last_light_vertex);
            let mat = world.get_material(last_light_vertex.material_id);
            // NOTE: wi and wo might need to be swapped, since the reference i was working from had them swapped in name.
            pdf_s_end_forward =
                mat.value(&as_hitrecord, direction, llv_wo_local).0 / llv_wo_local.z();
            pdf_s_end_backward =
                mat.value(&as_hitrecord, llv_wo_local, direction).0 / direction.z();
        }

        if t == 1 {
            // light path connecting to specific point on lens surface.
            // disallowed since film splatting (which requires recalculating the pixel) is not supported.
        } else {
            let second_to_last_eye_vertex = eye_path[t - 2];
            let last_eye_to_prev =
                (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
            let direction = lev_frame.to_local(&last_eye_to_prev);
            let as_hitrecord = HitRecord::from(last_eye_vertex);
            let mat = world.get_material(last_eye_vertex.material_id);
            // NOTE: wi and wo might need to be swapped, since the reference i was working from had them swapped in name.
            pdf_s_end_forward =
                mat.value(&as_hitrecord, direction, lev_wo_local).0 / lev_wo_local.z();
            pdf_s_end_backward =
                mat.value(&as_hitrecord, lev_wo_local, direction).0 / direction.z();
        }
    }

    let k = s + t - 1;
    if s > 0 {
        for i in 0..(s - 1) {
            // Light Vertex
            let lv = light_path[i];
            mis_nodes[i].pdf_toward_light = lv.pdf_backward * if i == 0 { 1.0 } else { lv.veach_g };
            mis_nodes[i].pdf_toward_eye = lv.pdf_forward * light_path[i + 1].veach_g;
            // mis_nodes[i].specular = lv.is_specular;
        }
        // Last Light Vertex
        let s1 = s - 1;
        let llv = light_path[s1];
        mis_nodes[s1].pdf_toward_light =
            pdf_s_end_backward * if s == 1 { 1.0 } else { llv.veach_g };
        mis_nodes[s1].pdf_toward_eye = pdf_s_end_forward * if s1 == k { 1.0 } else { veach_g };
        // mis_nodes[s-1].specular = llv.is_specular;
    }

    if t > 0 {
        for i in 0..(t - 1) {
            // Eye Vertex
            let ev = eye_path[i];
            let ki = k - i;
            mis_nodes[ki].pdf_toward_eye = ev.pdf_backward * if i == 0 { 1.0 } else { ev.veach_g };
            mis_nodes[ki].pdf_toward_light = ev.pdf_forward * eye_path[i + 1].veach_g;
            // mis_nodes[ki].specular = ev.is_specular;
        }
        let t1 = t - 1;
        let kt1 = k - t1;
        mis_nodes[kt1].pdf_toward_eye =
            pdf_t_end_backward * if t == 1 { 1.0 } else { eye_path[t - 1].veach_g };
        mis_nodes[kt1].pdf_toward_light = pdf_t_end_forward * if kt1 == 0 { 1.0 } else { veach_g }
        // mis_nodes[kt1].specular = eye_path[t-1].is_specular;
    }
    let mut p_k = 1.0;
    let mut mis_sum = 1.0;
    for i in s..(k + 1) {
        if i == 0 {
            p_k *= mis_nodes[0].pdf_toward_light / mis_nodes[1].pdf_toward_light;
            // if mid_nodes[1].specular {continue;}
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[0].pdf_toward_light,
                mis_nodes[1].pdf_toward_light
            );
        } else if i == k {
            /*if delta camera (i.e. pinhole or orthographic), break*/
            if mis_nodes[k].pdf_toward_eye == 0.0 {
                break;
            }
            p_k *= mis_nodes[k - 1].pdf_toward_eye / mis_nodes[k].pdf_toward_eye;
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[k - 1].pdf_toward_eye,
                mis_nodes[k].pdf_toward_eye
            );
        } else {
            p_k *= mis_nodes[i - 1].pdf_toward_eye / mis_nodes[i + 1].pdf_toward_light;
            /* if mis_nodes[i].specular || mis_nodes[i + 1].specular {
                continue;
            } */
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[i - 1].pdf_toward_eye,
                mis_nodes[i + 1].pdf_toward_light
            );
        }
        mis_sum += p_k * p_k;
    }
    // reset p_k and calculate again from the other side
    p_k = 1.0;
    for i in s..0 {
        if i == k + 1 {
            p_k *= mis_nodes[k].pdf_toward_eye / mis_nodes[k - 1].pdf_toward_eye;
            // if mis_nodes[k-1].specular {
            //     continue;
            // }
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[k].pdf_toward_eye,
                mis_nodes[k - 1].pdf_toward_eye
            );
        } else if i == 1 {
            // handle delta light
            // if world.get_material(light_path[0].material_id).is_delta() {break;}
            if mis_nodes[0].pdf_toward_light == 0.0 {
                break;
            }
            p_k *= mis_nodes[1].pdf_toward_light / mis_nodes[0].pdf_toward_light;
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[1].pdf_toward_light,
                mis_nodes[0].pdf_toward_light
            );
        } else {
            p_k *= mis_nodes[i].pdf_toward_light / mis_nodes[i - 2].pdf_toward_eye;
            assert!(
                !p_k.is_nan(),
                "{} {}",
                mis_nodes[i].pdf_toward_light,
                mis_nodes[i - 2].pdf_toward_eye
            );
            /* if mis_nodes[i-1].specular || mis_nodes[i -2].specular {
                continue;
            } */
        }
        mis_sum += p_k * p_k;
    }
    1.0 / mis_sum
}
