use crate::world::World;
// use crate::config::Settings;
use crate::hittable::{HasBoundingBox, HitRecord};
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;

use std::ops::Index;
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

#[derive(Debug, Copy, Clone, PartialEq)]
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

#[allow(unused_mut)]
pub fn random_walk(
    mut ray: Ray,
    lambda: f32,
    bounce_limit: u16,
    start_throughput: SingleEnergy,
    trace_type: Type,
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
                0.0,
                0.0,
                1.0,
            );

            let frame = TangentFrame::from_normal(hit.normal);
            let wi = frame.to_local(&-ray.direction).normalized();
            // let id: usize = hit.material.into();
            if let MaterialId::Camera(_camera_id) = hit.material {
                if trace_type == Type::Light {
                    vertex.kind = Type::Camera;
                    vertices.push(vertex);
                    break;
                }
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
                // if emission.0 > 0.0 {

                // }
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
                // last_bsdf_pdf = pdf;

                debug_assert!(!beta.0.is_nan(), "{:?} {} {:?}", f, cos_i, pdf);

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * 0.01 * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                // hit a surface and didn't bounce.
                if emission.0 > 0.0 {
                    vertex.kind = Type::LightSource(Source::Instance);
                }
                vertex.pdf_forward = 0.0;
                vertex.pdf_backward = 1.0;
                vertex.veach_g = veach_g(hit.point, 1.0, ray.origin, 1.0);
                vertices.push(vertex);
                break;
            }
        } else {
            // add a vertex when a camera ray hits the environment
            if trace_type == Type::Eye {
                let max_world_radius =
                    (world.bounding_box().max - world.bounding_box().min).norm() / 2.0;
                let max_world_radius_2 = max_world_radius * max_world_radius;
                assert!(max_world_radius.is_finite());
                let at_env = max_world_radius * ray.direction;
                assert!(at_env.0.is_finite().all());
                assert!(Point3::from(at_env).0.is_finite().all());
                let vertex = Vertex::new(
                    Type::LightSource(Source::Environment),
                    1.0 * max_world_radius,
                    lambda,
                    Point3::from(at_env),
                    -ray.direction,
                    MaterialId::Light(0),
                    0,
                    beta,
                    0.0,
                    1.0 / (max_world_radius_2 * 4.0 * PI),
                    1.0 / (max_world_radius_2),
                );
                assert!(vertex.point.0.is_finite().all());
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
        assert!(wi.0.is_finite().all(), "{:?}", eye_path);

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

pub struct CombinedPath<'a> {
    pub light_path: &'a Vec<Vertex>,
    pub eye_path: &'a Vec<Vertex>,
    pub connection_index: usize,
    pub path_length: usize,
}

impl<'a> Index<usize> for CombinedPath<'a> {
    type Output = Vertex;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.connection_index {
            let len = self.path_length;
            assert!(len >= index && index > 0);
            &self.eye_path[len - index]
        } else {
            assert!(index < self.light_path.len());
            &self.light_path[index]
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

    let k = s + t - 1;
    let k1 = k + 1;

    let combined_path = CombinedPath {
        light_path,
        eye_path,
        connection_index: s,
        path_length: k,
    };

    assert!(combined_path[0] == light_path[0]);
    assert!(combined_path[k] == eye_path[0]);
    assert!(combined_path[s] == light_path[s]);
    assert!(combined_path[s + 1] == light_path[k1 - s]);

    let mut path_ps: Vec<f32> = Vec::with_capacity(k1);
    path_ps.fill(0.0);
    path_ps[s] = 1.0;
    // notes about special cases:
    // for the eye subpath, vertex0.pdf_forward is 1.0 and vertex0.pdf_backward is 0.0 (cannot sample camera for now)
    // for the eye subpath, vertex1.pdf_forward is P_A, which technically is the directional pdf times the geometry term for the camera.
    // and pdf_backward is
    // for the light subpath, vertex0.pdf_forward
    // first build up from index = s to index = k
    for i in s..k1 {
        let i1 = i + 1;
        if i == 0 {
            // top case of equation 10.9
            // path_ps[1] = path_ps[0] * combined_path[0].pdf_forward / combined_path[0].pdf_backward;
        }
    }

    mis_function(&path_ps)
}
