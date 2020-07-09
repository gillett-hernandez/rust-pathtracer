// use crate::config::Settings;
// use crate::camera::*;
use crate::hittable::{HasBoundingBox, HitRecord};
use crate::integrator::veach_v;
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;
use crate::world::World;
use crate::{TransportMode, NORMAL_OFFSET};

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

impl From<TransportMode> for VertexType {
    fn from(value: TransportMode) -> Self {
        match value {
            TransportMode::Importance => VertexType::Eye,
            TransportMode::Radiance => VertexType::Light,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex {
    pub vertex_type: VertexType,
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
        vertex_type: VertexType,
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
            vertex_type,
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
        let transport_mode = match data.vertex_type {
            VertexType::Light | VertexType::LightSource(_) => TransportMode::Radiance,
            VertexType::Camera | VertexType::Eye => TransportMode::Importance,
        };
        HitRecord::new(
            data.time,
            data.point,
            (0.0, 0.0),
            data.lambda,
            data.normal,
            data.material_id,
            data.instance_id,
            Some(transport_mode),
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
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<Vertex>,
    russian_roulette_start_index: u16,
) -> Option<SingleEnergy> {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut additional_contribution = SingleEnergy::ZERO;
    // additional contributions from emission from hit objects that support bsdf sampling? review veach paper.
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.01, ray.tmax) {
            hit.lambda = lambda;
            hit.transport_mode = trace_type;
            let mut vertex = Vertex::new(
                trace_type.into(),
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

            if let MaterialId::Camera(_camera_id) = hit.material {
                if trace_type == TransportMode::Radiance {
                    // if hit camera directly while tracing a light path
                    vertex.vertex_type = VertexType::Camera;
                    vertices.push(vertex);
                }
                break;
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
                let f = material.f(&hit, wi, wo);
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                // if emission.0 > 0.0 {

                // }
                let mut pdf = material.value(&hit, wi, wo);
                debug_assert!(pdf.0 >= 0.0, "pdf was less than 0 {:?}", pdf);
                if pdf.0 < 0.00000001 || pdf.is_nan() {
                    break;
                }
                let rr_continue_prob = if bounce >= russian_roulette_start_index {
                    (f.0 / pdf.0).min(1.0)
                } else {
                    1.0
                };
                let russian_roulette_sample = sampler.draw_1d();
                if russian_roulette_sample.x > rr_continue_prob {
                    break;
                }
                beta *= f * cos_i.abs() / (rr_continue_prob * pdf.0);
                vertex.pdf_forward = rr_continue_prob * pdf.0 / cos_i;

                // consider handling delta distributions differently here, if deltas are ever added.
                // eval pdf in reverse direction
                vertex.pdf_backward = rr_continue_prob * material.value(&hit, wo, wi).0 / cos_o;

                debug_assert!(
                    vertex.pdf_forward > 0.0 && vertex.pdf_forward.is_finite(),
                    "pdf forward was 0 for material {:?} at vertex {:?}. wi: {:?}, wo: {:?}, cos_o: {}, cos_i: {}, rrcont={}",
                    material,
                    vertex,
                    wi,
                    wo,
                    cos_o,
                    cos_i,
                    rr_continue_prob,
                );
                debug_assert!(
                    vertex.pdf_backward > 0.0 && vertex.pdf_backward.is_finite(),
                    "pdf backward was 0 for material {:?} at vertex {:?}. wi: {:?}, wo: {:?}, cos_o: {}, cos_i: {}, rrcont={}",
                    material,
                    vertex,
                    wi,
                    wo,
                    cos_o,
                    cos_i,
                    rr_continue_prob,
                );

                vertices.push(vertex);

                // let beta_before_hit = beta;
                // last_bsdf_pdf = pdf;

                debug_assert!(!beta.0.is_nan(), "{:?} {:?} {} {:?}", beta.0, f, cos_i, pdf);

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * NORMAL_OFFSET * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                // hit a surface and didn't bounce.
                if emission.0 > 0.0 {
                    vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                    vertex.pdf_forward = 0.0;
                    vertex.pdf_backward = 1.0;
                    vertex.veach_g = veach_g(hit.point, wi.z().abs(), ray.origin, 1.0);
                    vertices.push(vertex);
                } else {
                    // this happens when the backside of a light is hit.
                }
                break;
            }
        } else {
            // add a vertex when a camera ray hits the environment
            if trace_type == TransportMode::Importance {
                let ray_direction = ray.direction;
                let bounding_box = world.aabb();
                let world_radius = (bounding_box.max - bounding_box.min).norm();
                let at_env = ray_direction * world_radius;
                let vertex = Vertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    ray.time,
                    lambda,
                    Point3::from(at_env),
                    -ray.direction,
                    MaterialId::Light(0),
                    0,
                    beta,
                    0.0,
                    1.0 / (4.0 * PI),
                    1.0,
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
    _sampler: &mut Box<dyn Sampler>,
    russian_roulette_threshold: f32,
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

    let mut cst: SingleEnergy;
    let mut sample = SampleKind::Sampled((SingleEnergy::ONE, 0.0));
    let g;
    if s == 0 {
        // since the eye path actually hit the light in this situation, calculate how much light would be transmitted along that eye path
        // consider resampling last_eye_vertex to be in a more favorable position.

        let second_to_last_eye_vertex = eye_path[t - 2];
        let last_eye_vertex = eye_path[t - 1];
        if last_eye_vertex.vertex_type == VertexType::LightSource(LightSourceType::Environment) {
            // second to last eye vertex is the one in the scene
            // last_eye_vertex is on env.
            let wo = last_eye_vertex.normal; // the env vertex stores the direction the ray was going
            let uv = direction_to_uv(wo);
            let emission = world.environment.emission(uv, last_eye_vertex.lambda);
            cst = emission;
            let cos_o = (wo * second_to_last_eye_vertex.normal).abs();
            g = cos_o;
        } else {
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
        }
    } else if t == 0 {
        let second_to_last_light_vertex = light_path[s - 2];
        let last_light_vertex = light_path[s - 1];
        if let MaterialId::Camera(camera_id) = last_light_vertex.material_id {
            let camera = world.get_camera(camera_id as usize);
            let direction = last_light_vertex.point - second_to_last_light_vertex.point;
            cst = SingleEnergy(
                camera
                    .eval_we(last_light_vertex.point, second_to_last_light_vertex.point)
                    .0,
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
        let mut ignore_distance_and_cos_o = false;

        let light_to_eye_vec = last_eye_vertex.point - last_light_vertex.point;
        let light_to_eye_direction = light_to_eye_vec.normalized();

        // llv means Last Light Vertex
        let llv_normal = last_light_vertex.normal;
        let llv_frame = TangentFrame::from_normal(llv_normal);
        let llv_world_light_to_eye = light_to_eye_direction;
        let llv_local_light_to_eye = llv_frame.to_local(&llv_world_light_to_eye).normalized();

        let fsl = if s == 1 {
            // connected to surface of light
            // consider resampling last_light_vertex to be in a more favorable position.

            if last_light_vertex.vertex_type
                == VertexType::LightSource(LightSourceType::Environment)
            {
                if last_eye_vertex.vertex_type
                    == VertexType::LightSource(LightSourceType::Environment)
                {
                    // can't connect an environment vertex to another environment vertex.
                    SingleEnergy::ZERO
                } else {
                    // connected to env vertex == llv, however since the point is some finite distance away (and not infinitely far like it would be for an env vertex)
                    // the direction would be off for a connection. so use the original direction that the env vertex would be coming in from in the calculations
                    // also make sure that veach_v is properly accounted for?
                    // maybe modify last_light_vertex's point to match the direction from last_eye_vertex
                    // ignore the actual Point3 location of the env vertex and instead use its direction as the connection direction
                    // also ignore the veach G term and only include the cosine, since the veach g term doesn't work with infinitely far rays.
                    let wo = -last_light_vertex.normal;
                    let uv = direction_to_uv(wo);
                    let emission = world.environment.emission(uv, last_eye_vertex.lambda);
                    ignore_distance_and_cos_o = true;
                    emission
                }
            } else {
                let hit_light_material = world.get_material(last_light_vertex.material_id);
                let emission = hit_light_material.emission(
                    &last_light_vertex.into(),
                    llv_local_light_to_eye,
                    None,
                );
                emission
            }
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
                SingleEnergy(
                    camera
                        .eval_we(last_eye_vertex.point, last_light_vertex.point)
                        .0,
                )
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
        if ignore_distance_and_cos_o {
            g = cos_i;
        } else {
            g = veach_g(last_eye_vertex.point, cos_i, last_light_vertex.point, cos_o);
        }
        if g == 0.0 {
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }

        let sample = _sampler.draw_1d().x;
        cst = fsl * g * fse;
        let russian_roulette_probability = (cst.0 / russian_roulette_threshold).min(1.0);
        if sample < russian_roulette_probability {
            if !veach_v(world, last_eye_vertex.point, last_light_vertex.point) {
                // not visible
                return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
            } else {
                cst *= 1.0 / russian_roulette_probability;
            }
        } else {
            return SampleKind::Sampled((SingleEnergy::ZERO, 0.0));
        }
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

#[derive(Debug)]
pub struct CombinedPath<'a> {
    pub light_path: &'a Vec<Vertex>,
    pub eye_path: &'a Vec<Vertex>,
    pub s: usize,
    pub t: usize,
    pub connecting_g: f32,
    pub light_vertex_pdf_forward: f32,
    pub light_vertex_pdf_backward: f32,
    pub eye_vertex_pdf_forward: f32,
    pub eye_vertex_pdf_backward: f32,
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
        // however on the eye side, the indices need to be modified.
        assert!(vidx0 + 1 == vidx1);
        assert!(self.s + self.t >= vidx0 + 1 || vidx1 < self.s);
        if vidx1 == self.s {
            // self.eye_path[t].veach_g
            // self.eye_path[self.path_length - vidx1].veach_g
            self.connecting_g
        } else if vidx1 < self.s {
            // if vidx1 is less than s, which is to say <= s-1
            self.light_path[vidx1].veach_g
        } else {
            // vidx1 must be > connection_index
            assert!(
                self.s + self.t >= vidx0 + 1,
                "mapped_index = {}, pl = {}, ci = {}, paths = {:?}",
                vidx0,
                self.s + self.t,
                self.s,
                self
            );
            let mapped_index = self.s + self.t - vidx0 - 1;
            assert!(
                mapped_index <= self.t,
                "mapped_index = {}, pl = {}, ci = {}, paths = {:?}",
                mapped_index,
                self.s + self.t,
                self.s,
                self
            );
            // let mapped_index = t - mapped_index;
            self.eye_path[mapped_index].veach_g
        }
    }

    pub fn pdf_forward(&self, vidx: usize) -> f32 {
        let vertex = self[vidx];
        let s = self.s;
        // if the request for the forward pdf is for the path connection, return it
        // forward connection pdf is the forward connection of index vidx == s-1;
        if vidx + 1 == s {
            // vidx == s - 1 is still a light vertex, but its probability is overridden
            self.light_vertex_pdf_forward
        } else if vidx == s {
            // vidx == s is the last eye vertex, but its probability is overridden
            self.eye_vertex_pdf_backward
        } else if vidx + 1 > s {
            // has to be swapped.
            vertex.pdf_backward
        } else {
            // vidx < s
            vertex.pdf_forward
        }
    }

    pub fn pdf_backward(&self, vidx: usize) -> f32 {
        let vertex = self[vidx];
        // if the request for the backward pdf is for the path connection, return it
        // backward connection pdf is the backward connection of index vidx == s;
        let s = self.s;
        if vidx + 1 == s {
            // vidx == s - 1 is still a light vertex, but its probability is overridden
            self.light_vertex_pdf_backward
        } else if vidx == s {
            // vidx == s is the last eye vertex, but its probability is overridden
            self.eye_vertex_pdf_forward
        } else if vidx + 1 > s {
            // has to be swapped.
            vertex.pdf_forward
        } else {
            // vidx < s
            vertex.pdf_backward
        }
    }
}

impl<'a> Index<usize> for CombinedPath<'a> {
    type Output = Vertex;
    fn index(&self, index: usize) -> &Self::Output {
        if index < self.s {
            debug_assert!(index < self.light_path.len());
            // println!("light: path index and subpath index {}", index);
            &self.light_path[index]
        } else {
            debug_assert!(
                self.s + self.t >= 1 + index,
                "index+1: {} >= path_length: {}",
                index + 1,
                self.s + self.t,
            );
            let index = self.s + self.t - index - 1;
            debug_assert!(index < self.eye_path.len());
            &self.eye_path[index]
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
    connecting_g: f32,
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

    // need to compute the connecting pdfs.

    let last_light_vertex = if s > 0 { Some(light_path[s - 1]) } else { None };
    let second_to_last_light_vertex = if s > 1 { Some(light_path[s - 2]) } else { None };

    let last_eye_vertex = if t > 0 { Some(eye_path[t - 1]) } else { None };
    let second_to_last_eye_vertex = if t > 1 { Some(eye_path[t - 2]) } else { None };

    // compute forward pdfs, which is lev to llv pdf and llv to lev pdf
    let mut llv_forward_pdf = 1.0f32;
    let mut lev_forward_pdf = 1.0f32;

    // recompute affected backward pdfs, that is, the backward pdfs of the vertices referred to as llv and lev, since their pdfs will have changed.
    let mut llv_backward_pdf = 1.0f32;
    let mut lev_backward_pdf = 1.0f32;

    // there are special cases need to be considered for t = 0 and s = 0. check the second arm of the following if branch
    if let (Some(llv), Some(lev)) = (last_light_vertex, last_eye_vertex) {
        let llv_normal = llv.normal;
        let lev_normal = lev.normal;

        let light_to_eye_vec = lev.point - llv.point;
        let light_to_eye_direction = light_to_eye_vec.normalized();
        let eye_to_light_vec = -light_to_eye_vec;
        let eye_to_light_direction = -light_to_eye_direction;

        let lev_frame = TangentFrame::from_normal(lev_normal);
        let llv_frame = TangentFrame::from_normal(llv_normal);

        let llv_world_light_to_eye = light_to_eye_direction;
        let llv_local_light_to_eye = llv_frame.to_local(&llv_world_light_to_eye).normalized();

        let lev_world_eye_to_light = eye_to_light_direction;
        let lev_local_eye_to_light = lev_frame.to_local(&lev_world_eye_to_light).normalized();

        // connecting forward pdf is the bsdf pdf supposing that there was a bsdf scatter from light_path[s-2] -> lightPath[s-1] -> eye_path[t-1]
        // however if s is 0, then connecting forward pdf will be the pdf of sampling the point and direction from the intersected light to eye_path[t-1]
        // additionally, if s = 1, then it is also a similar quantity. though this case can support resampling the light vertex light_path[0] to find a better vertex. perhaps that vertex should be passed in to this function

        if let Some(sllv) = second_to_last_light_vertex {
            let wi = (sllv.point - llv.point).normalized();
            let hit_material = world.get_material(llv.material_id);
            // let g = veach_g(
            //     lev.point,
            //     lev_local_eye_to_light.z().abs(),
            //     llv.point,
            //     llv_local_light_to_eye.z().abs(),
            // );
            llv_forward_pdf = hit_material
                .value(
                    &llv.into(),
                    llv_frame.to_local(&wi).normalized(),
                    llv_local_light_to_eye,
                )
                .0;
            llv_backward_pdf = hit_material
                .value(
                    &llv.into(),
                    llv_local_light_to_eye,
                    llv_frame.to_local(&wi).normalized(),
                )
                .0;
        } else {
            // s must be 1
            // which means the connection is to the surface of a light
            // if this case allows for resampling of the light surface vertex, account for the changed probability density here
            // llv is on surface of light
            // lev is in scene
            if llv.vertex_type == VertexType::LightSource(LightSourceType::Environment) {
                // second to last eye vertex is the one in the scene
                // llv is on env.
                // using same direction and uv for env, so no need to recompute pdf.
                let wo = (lev.point - llv.point).normalized();
                // let uv = direction_to_uv(wo);
                let g = (wo * lev.normal).abs();

                llv_forward_pdf = llv.pdf_forward;
                llv_backward_pdf = 0.0; // what to do here? since env probability would be expressed in solid angle space.
            } else {
                let hit_light_material = world.get_material(llv.material_id);

                // let normal = llv.normal;
                // let frame = TangentFrame::from_normal(normal);
                let wi = (lev.point - llv.point).normalized();
                debug_assert!(wi.0.is_finite().all(), "{:?}", eye_path);
                // let g = veach_g(
                //     lev.point,
                //     lev_local_eye_to_light.z().abs(),
                //     llv.point,
                //     llv_local_light_to_eye.z().abs(),
                // );
                llv_forward_pdf = hit_light_material
                    .emission_pdf(&llv.into(), llv_local_light_to_eye)
                    .0;
                llv_backward_pdf = 1.0; // put area sampling pdf here?
            }
        }

        // connecting backward pdf is the bsdf pdf supposing that there was a bsdf scatter from eye_path[t-2] -> eye_path[t-1] -> light_path[s-1]
        // however if t is 0, then connecting backward pdf will be the pdf of sampling the point and direction from the intersected camera pupil to light_path[s-1]
        // additionally, if t = 1, then it is also a similar quantity. though this case can support resampling the camera vertex eye_path[0] to find a better vertex. perhaps that vertex should be passed in to this function

        if let Some(slev) = second_to_last_eye_vertex {
            let wi = (slev.point - lev.point).normalized();
            let hit_material = world.get_material(lev.material_id);
            let llv_local_wi = lev_frame.to_local(&wi).normalized();
            // let g = veach_g(
            //     lev.point,
            //     lev_local_eye_to_light.z().abs(),
            //     llv.point,
            //     llv_local_light_to_eye.z().abs(),
            // );
            lev_forward_pdf = hit_material
                .value(&lev.into(), llv_local_wi, llv_local_light_to_eye)
                .0;
            lev_backward_pdf = hit_material
                .value(&lev.into(), llv_local_light_to_eye, llv_local_wi)
                .0;
        } else {
            // t must be 1
            // which means the connection is to the surface of the camera
            // if this case allows for resampling of the surface camera vertex, account for the changed probability density here
            if let MaterialId::Camera(camera_id) = lev.material_id {
                let g = veach_g(
                    lev.point,
                    lev_local_eye_to_light.z().abs(),
                    llv.point,
                    llv_local_light_to_eye.z().abs(),
                );
                let camera = world.get_camera(camera_id as usize);
                lev_forward_pdf = (camera.eval_we(lev.point, llv.point).1).0;
                lev_backward_pdf = 1.0; // do camera area sampling?
            } else {
                lev_forward_pdf = 0.0;
                lev_backward_pdf = 0.0;
            }
        }
    } else {
        // s == 0 or t == 0
        if s == 0 {
            // s = 0, which means that the eye path randomly intersected a light and that
            // lev (eye_path[t-1]) is on the surface of the light and
            // slev is in the scene
            let second_to_last_eye_vertex = eye_path[t - 2];
            let last_eye_vertex = eye_path[t - 1];
            if last_eye_vertex.vertex_type == VertexType::LightSource(LightSourceType::Environment)
            {
                // second to last eye vertex is the one in the scene
                // last_eye_vertex is on env.
                // using same direction and uv for env, so no need to recompute pdf.
                let wo = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
                // let uv = direction_to_uv(wo);
                // let cos_o = (wo * second_to_last_eye_vertex.normal).abs();

                llv_forward_pdf = last_eye_vertex.pdf_forward;
                llv_backward_pdf = 1.0; // env sampling pdf here maybe? maybe not since random sampling.
                lev_forward_pdf = 0.0;
                lev_backward_pdf = 0.0;
            } else {
                let hit_light_material = world.get_material(last_eye_vertex.material_id);

                let normal = last_eye_vertex.normal;
                let frame = TangentFrame::from_normal(normal);
                let wi = (second_to_last_eye_vertex.point - last_eye_vertex.point).normalized();
                debug_assert!(wi.0.is_finite().all(), "{:?}", eye_path);
                let (cos_i, cos_o) = (
                    (wi * normal).abs(), // these are cosines relative to their surface normals btw.
                    (wi * second_to_last_eye_vertex.normal).abs(), // i.e. eye_to_light.dot(eye_vertex_normal) and light_to_eye.dot(light_vertex_normal)
                );
                // let g = veach_g(
                //     last_eye_vertex.point,
                //     cos_i,
                //     second_to_last_eye_vertex.point,
                //     cos_o,
                // );
                llv_backward_pdf = hit_light_material
                    .emission_pdf(&last_eye_vertex.into(), frame.to_local(&wi).normalized())
                    .0;
                llv_forward_pdf = 1.0; // do light area sampling pdf
                lev_backward_pdf = 0.0;
                lev_forward_pdf = 0.0;
            }
        } else {
            // t = 0, which means that the eye path randomly intersected a camera and that
            // llv (light_path[t-1]) is on the surface of the camera and
            // sllv is in the scene
            let second_to_last_light_vertex = light_path[s - 2];
            let last_light_vertex = light_path[s - 1];
            if let MaterialId::Camera(camera_id) = last_light_vertex.material_id {
                let camera = world.get_camera(camera_id as usize);
                let direction = last_light_vertex.point - second_to_last_light_vertex.point;
                let normal = second_to_last_light_vertex.normal;
                // let g = veach_g(
                //     last_light_vertex.point,
                //     1.0,
                //     second_to_last_light_vertex.point,
                //     (normal * direction.normalized()).abs(),
                // );
                llv_forward_pdf = (camera
                    .eval_we(last_light_vertex.point, second_to_last_light_vertex.point)
                    .1)
                    .0;
                llv_backward_pdf = 1.0; // do camera area sampling pdf
                lev_forward_pdf = 0.0;
                lev_backward_pdf = 0.0;
            } else {
                llv_forward_pdf = 0.0;
                llv_backward_pdf = 0.0;
                lev_forward_pdf = 0.0;
                lev_backward_pdf = 0.0;
            }
        }
    }

    let path = CombinedPath {
        light_path,
        eye_path,
        s,
        t,
        connecting_g,
        light_vertex_pdf_forward: llv_forward_pdf,
        light_vertex_pdf_backward: llv_backward_pdf,
        eye_vertex_pdf_forward: lev_forward_pdf,
        eye_vertex_pdf_backward: lev_backward_pdf,
    };

    let mut ps: Vec<f32> = vec![0.0; s + t + 2];
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
                    connecting_g,
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

    // then calculate down from index = s-1 to index = 0
    for j in 1..s {
        let i = s - j; // i ranges from s to 1
        let ip1 = i + 1; // ip1 ranges from s+1 to 2
        let im1 = i - 1; // im1 ranges from s-1 to 0
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
                "i,s,t,k = ({}, {}, {}, {}). {:?} ===== {:?}",
                i,
                s,
                t,
                k,
                path[im1],
                path,
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
                connecting_g,
                path[i + 1],
                path.pdf_backward(ip1),
                veach_i_ip1,
                path.pdf_forward(im1),
                veach_im1_i
            );
        } else {
            // reciprocal top case of equation 10.9
            let pdf_forward = path.pdf_forward(0);
            let pdf_backward = path.pdf_backward(1);
            if pdf_forward == 0.0 {
                ps[0] = 0.0;
                continue;
            }
            // debug_assert!(
            //     path.pdf_forward(0) > 0.0,
            //     "i, s,t,k = ({}, {}, {}, {}). {:?}",
            //     i,
            //     s,
            //     t,
            //     k,
            //     path[0]
            // );
            ps[0] = ps[1] * pdf_backward * path.veach_g_between(0, 1) / pdf_forward;
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
