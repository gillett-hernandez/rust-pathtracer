use crate::hittable::{HasBoundingBox, HitRecord};
use crate::material::Material;
use crate::materials::MaterialId;
use crate::math::*;
use crate::world::World;
use crate::INTERSECTION_TIME_OFFSET;
use crate::{TransportMode, NORMAL_OFFSET};

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
    pub uv: (f32, f32),
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
        uv: (f32, f32),
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
            uv,
            material_id,
            instance_id,
            throughput,
            pdf_forward,
            pdf_backward,
            veach_g,
        }
    }

    pub fn default() -> Self {
        Vertex::new(
            VertexType::Eye,
            0.0,
            0.0,
            Point3::ORIGIN,
            Vec3::ZERO,
            (0.0, 0.0),
            MaterialId::Material(0),
            0,
            SingleEnergy::ZERO,
            0.0,
            0.0,
            0.0,
        )
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
            data.uv,
            data.lambda,
            data.normal,
            data.material_id,
            data.instance_id,
            Some(transport_mode),
        )
    }
}

pub fn veach_v(world: &Arc<World>, point0: Point3, point1: Point3) -> bool {
    // returns if the points are visible
    let diff = point1 - point0;
    let norm = diff.norm();
    let tmax = norm * 0.99;
    let point0_to_point1 = Ray::new_with_time_and_tmax(point0, diff / norm, 0.0, tmax);
    let hit = world.hit(point0_to_point1, INTERSECTION_TIME_OFFSET, tmax);
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

pub fn veach_g(point0: Point3, cos_i: f32, point1: Point3, cos_o: f32) -> f32 {
    (cos_i * cos_o).abs() / (point1 - point0).norm_squared()
}

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
                hit.uv,
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

            // consider accumulating emission in some other form for trace_type == TransportMode::Importance situations, as mentioned in veach.
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
                let pdf = material.value(&hit, wi, wo);
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
                    material.get_name(),
                    vertex,
                    wi,
                    wo,
                    cos_o,
                    cos_i,
                    rr_continue_prob,
                );
                // debug_assert!(
                //     vertex.pdf_backward >= 0.0 && vertex.pdf_backward.is_finite(),
                //     "pdf backward was 0 for material {:?} at vertex {:?}. wi: {:?}, wo: {:?}, cos_o: {}, cos_i: {}, rrcont={}",
                //     material.get_name(),
                //     vertex,
                //     wi,
                //     wo,
                //     cos_o,
                //     cos_i,
                //     rr_continue_prob,
                // );

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
                    (0.0, 0.0),
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
