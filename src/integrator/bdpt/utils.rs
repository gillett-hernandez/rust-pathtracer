use crate::world::World;
// use crate::config::Settings;
// use crate::camera::*;
use crate::hittable::{HasBoundingBox, HitRecord};
use crate::integrator::veach_v;
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;
use crate::NORMAL_OFFSET;

use std::ops::Index;
use std::sync::Arc;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LightSourceType {
    Instance,
    Environment,
}
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum VertexType {
    LightSource(LightSourceType),
    Light,
    Eye,
    Camera,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex {
    pub kind: VertexType,
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
        kind: VertexType,
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

#[allow(unused_mut)]
pub fn random_walk(
    mut ray: Ray,
    lambda: f32,
    bounce_limit: u16,
    start_throughput: SingleEnergy,
    trace_type: VertexType,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<Vertex>,
) -> Option<SingleEnergy> {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut additional_contribution = SingleEnergy::ZERO;
    // additional contributions from emission from hit objects that support bsdf sampling? review veach paper.
    for _ in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.01, ray.tmax) {
            hit.lambda = lambda;
            let mut vertex = Vertex::new(
                trace_type,
                hit.time,
                hit.lambda,
                hit.point,
                hit.normal,
                hit.material,
                hit.instance_id,
                beta,
                1.0,
                1.0,
                1.0,
            );

            let frame = TangentFrame::from_normal(hit.normal);
            let wi = frame.to_local(&-ray.direction).normalized();

            if trace_type == VertexType::Light {
                // if hit camera directly while tracing a light path
                if let MaterialId::Camera(_camera_id) = hit.material {
                    vertex.kind = VertexType::Camera;
                    vertices.push(vertex);
                    break;
                }
            } else {
                // if directly hit a light while tracing a camera path.
                if let MaterialId::Light(_light_id) = hit.material {}
            }

            let material = world.get_material(hit.material);

            // consider accumulating emission in some other form for trace_type == Type::Eye situations, as mentioned in veach.
            let maybe_wo: Option<Vec3> = material.generate(&hit, sampler.draw_2d(), wi);

            // what to do in this situation, where there is a wo and there's also emission?
            let emission = material.emission(&hit, wi, maybe_wo);

            // wo is generated in tangent space.

            if let Some(wo) = maybe_wo {
                // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                // if emission.0 > 0.0 {

                // }
                let pdf = material.value(&hit, wi, wo);
                debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                if pdf.0 < 0.00000001 || pdf.is_nan() {
                    break;
                }

                vertex.pdf_forward = pdf.0 / cos_i;

                if false && cos_o < 0.00001 {
                    // considered specular
                    vertex.pdf_backward = vertex.pdf_forward;
                } else {
                    // eval pdf in reverse direction
                    vertex.pdf_backward = material.value(&hit, wo, wi).0 / cos_o;
                }
                debug_assert!(
                    vertex.pdf_forward > 0.0 && vertex.pdf_forward.is_finite(),
                    "pdf forward was 0 for material {:?} at vertex {:?}",
                    material,
                    vertex
                );
                debug_assert!(
                    vertex.pdf_backward > 0.0 && vertex.pdf_backward.is_finite(),
                    "pdf backward was 0 for material {:?} at vertex {:?}. wi: {:?}, wo: {:?}, cos_o: {}, cos_i: {}",
                    material,
                    vertex,
                    wi,
                    wo,
                    cos_o,
                    cos_i
                );

                vertices.push(vertex);

                let f = material.f(&hit, wi, wo);

                // let beta_before_hit = beta;
                beta *= f * cos_i.abs() / pdf.0;
                // last_bsdf_pdf = pdf;

                debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * NORMAL_OFFSET * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                // hit a surface and didn't bounce.
                if emission.0 > 0.0 {
                    vertex.kind = VertexType::LightSource(LightSourceType::Instance);
                    vertex.pdf_forward = 0.0;
                    vertex.pdf_backward = 1.0;
                    vertex.veach_g = veach_g(hit.point, 1.0, ray.origin, 1.0);
                    vertices.push(vertex);
                } else {
                    // this happens when the backside of a light is hit.
                }
                break;
            }
        } else {
            // add a vertex when a camera ray hits the environment
            if trace_type == VertexType::Eye {
                let max_world_radius =
                    (world.bounding_box().max - world.bounding_box().min).norm() / 2.0;
                let max_world_radius_2 = max_world_radius * max_world_radius;
                debug_assert!(max_world_radius.is_finite());
                let at_env = max_world_radius * ray.direction;
                debug_assert!(at_env.0.is_finite().all());
                debug_assert!(Point3::from(at_env).0.is_finite().all());
                let vertex = Vertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    1.0 * max_world_radius,
                    lambda,
                    Point3::from(at_env),
                    -ray.direction,
                    MaterialId::Light(0),
                    0,
                    beta,
                    1.0 / (max_world_radius_2 * 4.0 * PI),
                    1.0,
                    1.0 / (max_world_radius_2),
                );
                debug_assert!(vertex.point.0.is_finite().all());
                // println!("sampling env and setting pdf_forward to 0");
                vertices.push(vertex);
            }
            break;
        }
    }
    if additional_contribution.0 > 0.0 {
        Some(additional_contribution)
    } else {
        None
    }
}

pub enum SampleKind {
    Sampled((SingleEnergy, f32)),
    Splatted((SingleEnergy, f32)),
}

pub fn eval_unweighted_contribution(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s: usize,
    eye_path: &Vec<Vertex>,
    t: usize,
) -> SampleKind {
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
    let mut sample = SampleKind::Sampled((SingleEnergy::ONE, 0.0));
    let g;
    if s == 0 {
        // since the eye path actually hit the light in this situation, calculate how much light would be transmitted along that eye path
        let second_to_last_eye_vertex = eye_path[t - 2];
        let last_eye_vertex = eye_path[t - 1];
        let hit_light_material = world.get_material(last_eye_vertex.material_id);

        let normal = last_eye_vertex.normal;
        let frame = TangentFrame::from_normal(normal);
        let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
        debug_assert!(wi.0.is_finite().all(), "{:?}", eye_path);
        let (cos_i, cos_o) = (
            (wi * normal).abs(), // these are cosines relative to their surface normals btw.
            (wi * second_to_last_eye_vertex.normal).abs(), // i.e. eye_to_light.dot(eye_vertex_normal) and light_to_eye.dot(light_vertex_normal)
        );
        g = veach_g(
            last_eye_vertex.point,
            cos_i,
            second_to_last_eye_vertex.point,
            cos_o,
        );
        cst = hit_light_material.emission(
            &last_eye_vertex.into(),
            frame.to_local(&wi).normalized(),
            None,
        );
    } else if t == 0 {
        let second_to_last_light_vertex = light_path[s - 2];
        let last_light_vertex = light_path[s - 1];
        if let MaterialId::Camera(camera_id) = last_light_vertex.material_id {
            let camera = world.get_camera(camera_id as usize);
            let direction = last_light_vertex.point - second_to_last_light_vertex.point;
            cst = SingleEnergy(
                camera.eval_we(last_light_vertex.point, second_to_last_light_vertex.point),
            );
            sample = SampleKind::Splatted((SingleEnergy::ONE, 0.0));

            let (cos_i, cos_o) = (
                (direction * second_to_last_light_vertex.normal).abs(), // these are cosines relative to their surface normals btw.
                (direction * last_light_vertex.normal).abs(), // i.e. eye_to_light.dot(eye_vertex_normal) and light_to_eye.dot(light_vertex_normal)
            );
            g = veach_g(
                last_light_vertex.point,
                cos_i,
                second_to_last_light_vertex.point,
                cos_o,
            );
        } else {
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }
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
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }

        // lev means Last Eye Vertex
        let lev_normal = last_eye_vertex.normal;
        let lev_frame = TangentFrame::from_normal(lev_normal);
        let lev_world_eye_to_light = -light_to_eye_direction;
        let lev_local_eye_to_light = lev_frame.to_local(&lev_world_eye_to_light).normalized();
        let fse = if t == 1 {
            // connected to surface of camera
            if let MaterialId::Camera(camera_id) = last_eye_vertex.material_id {
                let camera = world.get_camera(camera_id as usize);
                sample = SampleKind::Splatted((SingleEnergy::ONE, 0.0));
                SingleEnergy(camera.eval_we(last_eye_vertex.point, last_light_vertex.point))
            } else {
                SingleEnergy(0.0)
            }
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
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }

        let (cos_i, cos_o) = (
            lev_local_eye_to_light.z().abs(), // these are cosines relative to their surface normals btw.
            llv_local_light_to_eye.z().abs(), // i.e. eye_to_light.dot(eye_vertex_normal) and light_to_eye.dot(light_vertex_normal)
        );
        g = veach_g(last_eye_vertex.point, cos_i, last_light_vertex.point, cos_o);
        if g == 0.0 {
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }

        if !veach_v(world, last_eye_vertex.point, last_light_vertex.point) {
            // not visible
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }
        cst = fsl * g * fse;
    }
    match sample {
        SampleKind::Sampled(_) => SampleKind::Sampled((
            cst * last_eye_vertex_throughput * last_light_vertex_throughput,
            g,
        )),
        SampleKind::Splatted(_) => SampleKind::Splatted((
            cst * last_light_vertex_throughput * last_eye_vertex_throughput,
            g,
        )),
    }
}

pub struct CombinedPath<'a> {
    pub light_path: &'a Vec<Vertex>,
    pub eye_path: &'a Vec<Vertex>,
    pub connection_index: usize,
    pub connecting_g: f32,
    pub path_length: usize,
}

impl<'a> CombinedPath<'a> {
    pub fn veach_g_between(&self, vidx0: usize, vidx1: usize) -> f32 {
        // computes the veach g term between vertex v0 and v1.
        // veach g terms are stored flowing away from the endpoints, however for the combined path we want them flowing away from 0.
        // i.e. for the normal paths, eye_path[i] contains the veach g term for eye_path[i] to eye_path[i-1]
        // and for the normal paths, light_path[i] contains the veach g term for light_path[i] to light_path[i-1]
        // but we want combined_path[i].veach_g to be between combined_path[i] and combined_path[i-1] at all times

        // naive version: recompute veach g term

        // smart version
        // veach g is already computed and stored in vertices, only the order is reversed.
        // for both the eye and light subpaths, vertex[i].veach_g is the G term between y[i] and y[i-1] or z[i] and z[i-1]
        // however our indices are switched, and x_i.veach_g needs to be between x_i-1 and x_i
        // on the light side, the indices don't need to be switched
        // however on the eye side, the indices need to be switched slightly.
        if vidx1 >= self.connection_index && vidx0 < self.connection_index {
            // self.eye_path[t].veach_g
            // self.eye_path[self.path_length - vidx1].veach_g
            self.connecting_g
        } else if vidx1 >= self.connection_index {
            self.eye_path[self.path_length - vidx1 - 1].veach_g
        } else {
            self.light_path[vidx1].veach_g
        }
    }

    pub fn pdf_forward(&self, vidx: usize) -> f32 {
        let vertex = self[vidx];
        if vidx >= self.connection_index {
            // has to be swapped.
            vertex.pdf_backward
        } else {
            vertex.pdf_forward
        }
    }

    pub fn pdf_backward(&self, vidx: usize) -> f32 {
        let vertex = self[vidx];
        if vidx >= self.connection_index {
            // has to be swapped.
            vertex.pdf_forward
        } else {
            vertex.pdf_backward
        }
    }
}

impl<'a> Index<usize> for CombinedPath<'a> {
    type Output = Vertex;
    fn index(&self, index: usize) -> &Self::Output {
        if index < self.connection_index {
            debug_assert!(index < self.light_path.len());
            // println!("light: path index and subpath index {}", index);
            &self.light_path[index]
        } else {
            &self.eye_path[self.path_length - index - 1]
        }
    }
}

#[allow(unused)]
pub fn eval_mis<F>(
    world: &Arc<World>,
    light_path: &Vec<Vertex>,
    s: usize,
    eye_path: &Vec<Vertex>,
    t: usize,
    veach_g: f32,
    mis_function: F,
) -> f32
where
    F: FnOnce(&Vec<f32>) -> f32,
{
    // computes the mis weight of generating the path determined by s and t
    // path index is i = s
    // need to compute the relative probabilities of all the other paths that have the same path length
    // can be done recursively from the path probability of index = s by setting that probability to 1
    // and recursively/iteratively applying a scaling factor depending on the direction (towards the light root or towards the camera root)
    // the scaling factor towards the light root is p_i+1 / p_i
    // refer to veach (1997) page 306 equation 10.9

    let k = s + t - 1; // for 2,0 case, k is 1
    let k1 = k + 1; // k1 is 2

    // if t == 0 {
    //     // hit camera directly, would have caused index error.
    //     // for now, return 0
    //     println!("{:?} ={:?}= {:?}", light_path, veach_g, eye_path);
    //     return 1.0;
    // }
    if s + t == 2 {
        return 1.0;
    }

    // general notes:

    // a light subpath of length s = 5 is indexed from 0 to s-1, or 0 to 4. [0, 1, 2, 3, 4]
    // a eye subpath of length t = 3 is index from 0 to t-1, or from 0 to 2. [0, 1, 2]
    // so a combined path should only ever index into its componenet paths by s-1 for a light subpath, or t-1 for an eye subpath.
    // for k1 = s + t, the maximal index of a light subpath is connection_index - 1, and the maximal index of a eye subpath is k1 - connection_index - 1
    // in general, those are the bounds that should be checked. inclusive checking would be <= connection_index -1, but exclusive checking may be used as well
    // exclusive checking would be < connection_index for light subpaths, and < k1 - connection_index for eye subpath
    // for some generic index i then, if i < connection_index, it should be used to access the light subpath,
    // however on the other hand, we can define a new number to index into the eye subpath as j = k1 - i - 1
    // eye subpath should be indexed in 0 <= some_index < k1 - connection_index - 1
    // and should be indexed from furthest from the root to closest to the root
    // i.e. combined_path[s] should be eye_path[t-1]
    // and combined_path[s-1] should be light_path[s-1]
    // and combined_path[k = s + t - 1] should be eye_path[0]
    // thus 0 == N - (s + t - 1)
    // => 0 == N - s - t + 1
    // => N = s + t - 1
    // => N = k
    // the new_index should then be k - index
    // for index = s + 1, then new_index = k - (s + 1) = s + t - 1 - s - 1 = t - 1, which matches what is desired

    let path = CombinedPath {
        light_path,
        eye_path,
        connection_index: s,
        connecting_g: veach_g,
        path_length: k1,
    };

    let mut ps: Vec<f32> = vec![0.0; s + t + 1];
    ps[s] = 1.0;

    // first build up from index = s to index = k
    for i in s..=k {
        let ip1 = i + 1;
        if i == 0 {
            // top case of equation 10.9

            debug_assert!(
                path.pdf_backward(1) > 0.0,
                "i,s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[1]
            );
            ps[1] =
                ps[0] * path.pdf_forward(0) / (path.pdf_backward(1) * path.veach_g_between(0, 1));
            debug_assert!(!ps[1].is_nan(), "{:?}", ps);
        } else if i < k {
            let im1 = i - 1;
            // middle case of equation 10.9
            debug_assert!(
                path.pdf_backward(ip1) > 0.0,
                "i,s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[ip1]
            );
            let veach_im1_i = path.veach_g_between(im1, i);
            let veach_i_ip1 = path.veach_g_between(i, ip1);
            ps[ip1] = ps[i] * path.pdf_forward(im1) * veach_im1_i
                / (path.pdf_backward(ip1) * veach_i_ip1);
            debug_assert!(
                    !ps[ip1].is_nan(),
                    "path probabilities: {:?}, path[i]: {:?}, veach_G between: {:?}, path[i+1]: {:?}\n{:?} * {:?} / ({:?} * {:?})",
                    ps,
                    path[i],
                    veach_g,
                    path[i + 1],
                    path.pdf_forward(im1),
                    veach_im1_i,
                    path.pdf_backward(ip1),
                    veach_i_ip1,
                );
        } else {
            // bottom case of equation 10.9
            debug_assert!(
                path.pdf_backward(k) > 0.0,
                "i,s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[k]
            );
            ps[k1] = ps[k] * path.pdf_forward(k - 1) * path.veach_g_between(k - 1, k)
                / path.pdf_backward(k);
            debug_assert!(!ps[k1].is_nan(), "{:?}", ps);
        }
    }

    for j in 0..s {
        let i = s - j;
        let ip1 = i + 1;
        let im1 = i - 1;
        if i == k {
            // reciprocal bottom case of equation 10.9
            debug_assert!(
                path.pdf_forward(k - 1) > 0.0,
                "i,s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[k - 1]
            );
            ps[k] = ps[k1] * path.pdf_backward(k)
                / (path.pdf_forward(k - 1) * path.veach_g_between(k - 1, k));
            debug_assert!(!ps[k].is_nan(), "{:?}", ps);
        } else if i > 1 {
            // reciprocal middle case of equation 10.9
            debug_assert!(
                path.pdf_forward(im1) > 0.0,
                "i,s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[im1]
            );
            let veach_i_ip1 = path.veach_g_between(i, ip1);
            let veach_im1_i = path.veach_g_between(im1, i);
            ps[i] = ps[ip1] * path.pdf_backward(ip1) * veach_i_ip1
                / (path.pdf_forward(im1) * veach_im1_i);
            debug_assert!(
                !ps[i].is_nan(),
                "path probabilities: {:?}, path[i]: {:?}, veach_G between: {:?}, path[i+1]: {:?}\n{:?} * {:?} / ({:?} * {:?})",
                ps,
                path[i],
                veach_g,
                path[i + 1],
                path.pdf_backward(ip1),
                veach_i_ip1,
                path.pdf_forward(im1),
                veach_im1_i
            );
        } else {
            // reciprocal top case of equation 10.9
            debug_assert!(
                path.pdf_forward(0) > 0.0,
                "i, s,t,k = ({}, {}, {}, {}). {:?}",
                i,
                s,
                t,
                k,
                path[0]
            );
            ps[0] = ps[1] * path.pdf_backward(1) * path.veach_g_between(0, 1) / path.pdf_forward(0);
            debug_assert!(!ps[0].is_nan(), "{:?}", ps);
        }
    }

    for p in ps.iter() {
        debug_assert!(p.is_finite() && !p.is_nan(), "{:?}", ps);
    }
    let result = mis_function(&ps);
    debug_assert!(result.is_finite());
    result
}
/*
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mis_weights_for_short_paths() {
        // [
        //      Vertex { kind: LightSource(Instance), time: 0.0, lambda: 603.45825, point: Point3(f32x4(-0.027535105, -0.19972621, 0.9, 1.0)), normal: Vec3(f32x4(0.0, 0.0, -1.0, 0.0)), material_id: Light(3), instance_id: 5, throughput: SingleEnergy(3.0513232), pdf_forward: 0.31831563, pdf_backward: 1.9894367, veach_g: 1.0 },
        //      Vertex { kind: Camera, time: 5.0592623, lambda: 603.45825, point: Point3(f32x4(-5.0, -0.007217303, -0.013055563, 1.0)), normal: Vec3(f32x4(1.0, -0.0, -0.0, -0.0)), material_id: Camera(0), instance_id: 10, throughput: SingleEnergy(3.0513232), pdf_forward: 1.0, pdf_backward: 1.0, veach_g: 1.0 }
        // ]

        // [
        //      Vertex { kind: Camera, time: 0.59899455, lambda: 603.45825, point: Point3(f32x4(-5.0, 0.013205296, -0.0029486567, 1.0)), normal: Vec3(f32x4(0.98173434, -0.06221859, -0.17979613, 0.0)), material_id: Camera(0), instance_id: 0, throughput: SingleEnergy(1.0), pdf_forward: 1.0, pdf_backward: 0.01, veach_g: 1.0 },
        //      Vertex { kind: Eye, time: 4.5407047, lambda: 603.45825, point: Point3(f32x4(-0.5422344, -0.26931095, -0.81934977, 1.0)), normal: Vec3(f32x4(-0.80744135, 0.4356266, -0.3978293, 0.0)), material_id: Material(4), instance_id: 7, throughput: SingleEnergy(1.0), pdf_forward: 69214.664, pdf_backward: 10360.568, veach_g: 0.035089824 }
        // ]
    }
}*/
