use crate::hittable::HitRecord;
use crate::materials::{Material, MaterialId};
use crate::math::*;
use crate::mediums::Medium;
use crate::profile::Profile;
use crate::world::World;
use crate::world::INTERSECTION_TIME_OFFSET;
use crate::world::{TransportMode, NORMAL_OFFSET};

use std::{ops::AddAssign, sync::Arc};

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
pub struct SurfaceVertex {
    pub vertex_type: VertexType,
    pub time: f32,
    pub lambda: f32,
    pub local_wi: Vec3,
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

impl SurfaceVertex {
    pub fn new(
        vertex_type: VertexType,
        time: f32,
        lambda: f32,
        local_wi: Vec3,
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
        SurfaceVertex {
            vertex_type,
            time,
            lambda,
            local_wi,
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
        SurfaceVertex::new(
            VertexType::Eye,
            0.0,
            0.0,
            Vec3::ZERO,
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

impl From<SurfaceVertex> for HitRecord {
    fn from(data: SurfaceVertex) -> Self {
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

#[allow(unused_mut)]
pub fn random_walk(
    mut ray: Ray,
    lambda: f32,
    bounce_limit: u16,
    start_throughput: SingleEnergy,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<SurfaceVertex>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
) -> Option<SingleEnergy> {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut additional_contribution = SingleEnergy::ZERO;
    // additional contributions from emission from hit objects that support bsdf sampling? review veach paper.
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.01, ray.tmax) {
            hit.lambda = lambda;
            hit.transport_mode = trace_type;
            let mut vertex = SurfaceVertex::new(
                trace_type.into(),
                hit.time,
                hit.lambda,
                -ray.direction,
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
            let maybe_wo: Option<Vec3> = material.generate(
                hit.lambda,
                hit.uv,
                hit.transport_mode,
                sampler.draw_2d(),
                wi,
            );

            // what to do in this situation, where there is a wo and there's also emission?
            let emission = material.emission(hit.lambda, hit.uv, hit.transport_mode, wi);

            // wo is generated in tangent space.

            if let Some(wo) = maybe_wo {
                // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                let (f, pdf) = material.bsdf(hit.lambda, hit.uv, hit.transport_mode, wi, wo);
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                // if emission.0 > 0.0 {

                // }

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
                vertex.pdf_backward = rr_continue_prob
                    * material
                        .bsdf(hit.lambda, hit.uv, hit.transport_mode, wo, wi)
                        .1
                         .0
                    / cos_o;

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
                let world_radius = world.get_world_radius();
                let at_env = ray_direction * world_radius;
                let vertex = SurfaceVertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    ray.time,
                    lambda,
                    ray.direction,
                    Point3::from(at_env),
                    ray.direction,
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
    profile.bounce_rays += vertices.len();

    if additional_contribution.0 > 0.0 {
        Some(additional_contribution)
    } else {
        None
    }
}
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct HeroEnergy(pub f32x4);

impl HeroEnergy {
    pub const ZERO: Self = HeroEnergy(f32x4::splat(0.0));
}

impl AddAssign for HeroEnergy {
    fn add_assign(&mut self, rhs: HeroEnergy) {
        self.0 += rhs.0;
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct HeroSurfaceVertex {
    pub vertex_type: VertexType,
    pub time: f32,
    pub lambda: f32x4,
    pub local_wi: Vec3,
    pub point: Point3,
    pub normal: Vec3,
    pub uv: (f32, f32),
    pub material_id: MaterialId,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
    pub instance_id: usize,
    pub throughput: HeroEnergy,
    pub pdf_forward: f32x4,
    pub pdf_backward: f32x4,
    pub veach_g: f32,
}

impl HeroSurfaceVertex {
    pub fn new(
        vertex_type: VertexType,
        time: f32,
        lambda: f32x4,
        local_wi: Vec3,
        point: Point3,
        normal: Vec3,
        uv: (f32, f32),
        material_id: MaterialId,
        instance_id: usize,
        throughput: HeroEnergy,
        pdf_forward: f32x4,
        pdf_backward: f32x4,
        veach_g: f32,
    ) -> Self {
        HeroSurfaceVertex {
            vertex_type,
            time,
            lambda,
            local_wi,
            point,
            normal,
            uv,
            material_id,
            outer_medium_id: 0,
            inner_medium_id: 0,
            instance_id,
            throughput,
            pdf_forward,
            pdf_backward,
            veach_g,
        }
    }

    pub fn default() -> Self {
        HeroSurfaceVertex::new(
            VertexType::Eye,
            0.0,
            f32x4::splat(0.0),
            Vec3::ZERO,
            Point3::ORIGIN,
            Vec3::ZERO,
            (0.0, 0.0),
            MaterialId::Material(0),
            0,
            HeroEnergy::ZERO,
            f32x4::splat(0.0),
            f32x4::splat(0.0),
            0.0,
        )
    }
    pub fn transport_mode(&self) -> TransportMode {
        match self.vertex_type {
            VertexType::Light | VertexType::LightSource(_) => TransportMode::Radiance,
            VertexType::Camera | VertexType::Eye => TransportMode::Importance,
        }
    }
    pub fn into_hit_w_lane(&self, lane: usize) -> HitRecord {
        HitRecord::new(
            self.time,
            self.point,
            self.uv,
            self.lambda.extract(lane),
            self.normal,
            self.material_id,
            self.instance_id,
            Some(self.transport_mode()),
        )
    }
}

#[allow(unused_mut)]
pub fn random_walk_hero(
    mut ray: Ray,
    lambda: f32x4,
    bounce_limit: u16,
    start_throughput: f32x4,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<HeroSurfaceVertex>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
) -> Option<HeroEnergy> {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut additional_contribution = HeroEnergy::ZERO;
    // additional contributions from emission from hit objects that support bsdf sampling? review veach paper.
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.01, ray.tmax) {
            hit.lambda = lambda.extract(0);
            hit.transport_mode = trace_type;
            let mut vertex = HeroSurfaceVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                -ray.direction,
                hit.point,
                hit.normal,
                hit.uv,
                hit.material,
                hit.instance_id,
                HeroEnergy(beta),
                f32x4::splat(1.0),
                f32x4::splat(1.0),
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
            let maybe_wo: Option<Vec3> = material.generate(
                hit.lambda,
                hit.uv,
                hit.transport_mode,
                sampler.draw_2d(),
                wi,
            );

            // what to do in this situation, where there is a wo and there's also emission?
            let multi_emission = HeroEnergy(f32x4::new(
                material
                    .emission(lambda.extract(0), hit.uv, hit.transport_mode, wi)
                    .0,
                material
                    .emission(lambda.extract(1), hit.uv, hit.transport_mode, wi)
                    .0,
                material
                    .emission(lambda.extract(2), hit.uv, hit.transport_mode, wi)
                    .0,
                material
                    .emission(lambda.extract(3), hit.uv, hit.transport_mode, wi)
                    .0,
            ));

            // wo is generated in tangent space.

            if let Some(wo) = maybe_wo {
                // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                let (multi_f, multi_pdf) = {
                    let (f0, pdf0) =
                        material.bsdf(lambda.extract(0), hit.uv, hit.transport_mode, wi, wo);
                    let (f1, pdf1) =
                        material.bsdf(lambda.extract(1), hit.uv, hit.transport_mode, wi, wo);
                    let (f2, pdf2) =
                        material.bsdf(lambda.extract(2), hit.uv, hit.transport_mode, wi, wo);
                    let (f3, pdf3) =
                        material.bsdf(lambda.extract(3), hit.uv, hit.transport_mode, wi, wo);
                    (
                        f32x4::new(f0.0, f1.0, f2.0, f3.0),
                        f32x4::new(pdf0.0, pdf1.0, pdf2.0, pdf3.0),
                    )
                };
                let (_reverse_multi_f, reverse_multi_pdf) = {
                    let (f0, pdf0) =
                        material.bsdf(lambda.extract(0), hit.uv, hit.transport_mode, wo, wi);
                    let (f1, pdf1) =
                        material.bsdf(lambda.extract(1), hit.uv, hit.transport_mode, wo, wi);
                    let (f2, pdf2) =
                        material.bsdf(lambda.extract(2), hit.uv, hit.transport_mode, wo, wi);
                    let (f3, pdf3) =
                        material.bsdf(lambda.extract(3), hit.uv, hit.transport_mode, wo, wi);
                    (
                        f32x4::new(f0.0, f1.0, f2.0, f3.0),
                        f32x4::new(pdf0.0, pdf1.0, pdf2.0, pdf3.0),
                    )
                };
                let cos_i = wo.z().abs();
                let cos_o = wi.z().abs();
                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                // if emission.0 > 0.0 {

                // }

                let hero_f = multi_f.extract(0);
                let hero_pdf = multi_pdf.extract(0);
                debug_assert!(hero_pdf >= 0.0, "pdf was less than 0 {:?}", hero_pdf);
                if hero_pdf < 0.00000001 || hero_pdf.is_nan() {
                    break;
                }
                let rr_continue_prob = if bounce >= russian_roulette_start_index {
                    (hero_f / hero_pdf).min(1.0)
                } else {
                    1.0
                };
                let russian_roulette_sample = sampler.draw_1d();
                if russian_roulette_sample.x > rr_continue_prob {
                    break;
                }
                beta *= multi_f * cos_i.abs() / (rr_continue_prob * hero_pdf);
                vertex.pdf_forward = rr_continue_prob * multi_pdf / cos_i;

                // consider handling delta distributions differently here, if deltas are ever added.
                // eval pdf in reverse direction
                vertex.pdf_backward = rr_continue_prob * reverse_multi_pdf / cos_o;

                debug_assert!(
                    vertex.pdf_forward.extract(0) > 0.0 && vertex.pdf_forward.is_finite().all(),
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

                debug_assert!(
                    !beta.extract(0).is_nan(),
                    "{:?} {:?} {} {:?}",
                    beta,
                    multi_f,
                    cos_i,
                    multi_pdf
                );

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * NORMAL_OFFSET * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                    frame.to_world(&wo).normalized(),
                );
            } else {
                // hit a surface and didn't bounce.
                if multi_emission.0.gt(f32x4::splat(0.0)).any() {
                    vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                    vertex.pdf_forward = f32x4::splat(0.0);
                    vertex.pdf_backward = f32x4::splat(1.0);
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
                let world_radius = world.get_world_radius();
                let at_env = ray_direction * world_radius;
                let vertex = HeroSurfaceVertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    ray.time,
                    lambda,
                    ray.direction,
                    Point3::from(at_env),
                    ray.direction,
                    (0.0, 0.0),
                    MaterialId::Light(0),
                    0,
                    HeroEnergy(beta),
                    f32x4::splat(0.0),
                    f32x4::splat(1.0 / (4.0 * PI)),
                    1.0,
                );
                debug_assert!(vertex.point.0.is_finite().all());
                // println!("sampling env and setting pdf_forward to 0");
                vertices.push(vertex);
            }
            break;
        }
    }
    profile.bounce_rays += vertices.len();

    if additional_contribution.0.gt(f32x4::splat(0.0)).any() {
        Some(additional_contribution)
    } else {
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct HeroMediumVertex {
    pub vertex_type: VertexType,
    pub time: f32,
    pub lambda: f32x4,
    pub wi: Vec3,
    pub point: Point3,
    pub uvw: (f32, f32, f32),
    pub medium_id: usize,
    pub instance_id: usize,
    pub throughput: HeroEnergy,
    pub pdf_forward: f32x4,
    pub pdf_backward: f32x4,
    pub veach_g: f32,
}

impl HeroMediumVertex {
    pub fn new(
        vertex_type: VertexType,
        time: f32,
        lambda: f32x4,
        wi: Vec3,
        point: Point3,
        uvw: (f32, f32, f32),
        medium_id: usize,
        instance_id: usize,
        throughput: HeroEnergy,
        pdf_forward: f32x4,
        pdf_backward: f32x4,
        veach_g: f32,
    ) -> Self {
        HeroMediumVertex {
            vertex_type,
            time,
            lambda,
            wi,
            point,
            uvw,
            medium_id,
            instance_id,
            throughput,
            pdf_forward,
            pdf_backward,
            veach_g,
        }
    }

    pub fn default() -> Self {
        HeroMediumVertex::new(
            VertexType::Eye,
            0.0,
            f32x4::splat(0.0),
            Vec3::ZERO,
            Point3::ORIGIN,
            (0.0, 0.0, 0.0),
            0,
            0,
            HeroEnergy::ZERO,
            f32x4::splat(0.0),
            f32x4::splat(0.0),
            0.0,
        )
    }
    pub fn transport_mode(&self) -> TransportMode {
        match self.vertex_type {
            VertexType::Light | VertexType::LightSource(_) => TransportMode::Radiance,
            VertexType::Camera | VertexType::Eye => TransportMode::Importance,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum HeroVertex {
    Surface(HeroSurfaceVertex),
    Medium(HeroMediumVertex),
}

impl HeroVertex {
    pub fn point(&self) -> Point3 {
        match self {
            HeroVertex::Medium(v) => v.point,
            HeroVertex::Surface(v) => v.point,
        }
    }
    pub fn pdf_forward(&self) -> f32x4 {
        match self {
            HeroVertex::Medium(v) => v.pdf_forward,
            HeroVertex::Surface(v) => v.pdf_forward,
        }
    }
    pub fn cos(&self, vec: Vec3) -> f32 {
        match self {
            HeroVertex::Medium(_v) => 1.0,
            HeroVertex::Surface(v) => v.normal * vec,
        }
    }
}

#[allow(unused_mut)]
pub fn random_walk_medium_hero(
    mut ray: Ray,
    lambda: f32x4,
    bounce_limit: u16,
    start_throughput: f32x4,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<HeroVertex>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
) -> Option<HeroEnergy> {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut additional_contribution = HeroEnergy::ZERO;
    // additional contributions from emission from hit objects that support bsdf sampling? review veach paper.
    let mut tracked_mediums: Vec<usize> = Vec::new();
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.001, ray.tmax) {
            hit.lambda = lambda.extract(0);
            hit.transport_mode = trace_type;
            let mut surface_vertex = HeroSurfaceVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                -ray.direction,
                hit.point,
                hit.normal,
                hit.uv,
                hit.material,
                hit.instance_id,
                HeroEnergy(beta),
                f32x4::splat(1.0),
                f32x4::splat(1.0),
                1.0,
            );

            let mut medium_vertex = HeroMediumVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                -ray.direction,
                hit.point,
                (0.0, 0.0, 0.0),
                0,
                hit.instance_id,
                HeroEnergy(beta),
                f32x4::splat(1.0),
                f32x4::splat(1.0),
                1.0,
            );

            let mut vertex = HeroVertex::Surface(surface_vertex);

            let mut hero_weight = 1.0;
            let mut hero_tr = 1.0;
            for medium_id in tracked_mediums.iter() {
                let medium = &world.mediums[*medium_id - 1];
                let (p, tr, scatter) =
                    medium.sample(lambda.extract(0), ray, Sample1D::new_random_sample());
                if scatter {
                    let t = (p - ray.origin).norm();
                    if t < medium_vertex.time {
                        medium_vertex.time = t;
                        medium_vertex.point = p;
                        hero_weight = tr;
                        hero_tr = medium.tr(lambda.extract(0), ray.origin, p);
                        medium_vertex.medium_id = *medium_id;
                        // println!(
                        //     "overrode surface vertex with medium vertex, p = {:?}",
                        //     medium_vertex.point
                        // );
                        vertex = HeroVertex::Medium(medium_vertex);
                    }
                }
            }
            // multiply in hero weight, since it includes some of the hero pdf information and that would be lost if unaccounted for.
            // hero weight also includes tr.
            beta *= hero_weight;
            for i in 0..4 {
                let mut combined_throughput = 1.0;
                for medium_id in tracked_mediums.iter() {
                    if *medium_id == medium_vertex.medium_id && i == 0 {
                        // skip hero
                        continue;
                    }
                    let medium = &world.mediums[*medium_id - 1];
                    combined_throughput *=
                        medium.tr(lambda.extract(i), ray.origin, medium_vertex.point);
                }

                // divide out hero_tr for all wavelengths, since it was included in overall beta mult.
                beta = beta.replace(i, beta.extract(i) * combined_throughput / hero_tr);
            }
            // multiply hero_tr back in for only hero wavelength.
            beta = beta.replace(0, beta.extract(0) * hero_tr);

            match vertex {
                HeroVertex::Surface(mut vertex) => {
                    let frame = TangentFrame::from_normal(hit.normal);
                    let wi = frame.to_local(&-ray.direction).normalized();

                    if let MaterialId::Camera(_camera_id) = hit.material {
                        if trace_type == TransportMode::Radiance {
                            // if hit camera directly while tracing a light path
                            surface_vertex.vertex_type = VertexType::Camera;
                            vertices.push(HeroVertex::Surface(surface_vertex));
                        }
                        break;
                    } else {
                        // if directly hit a light while tracing a camera path.
                        if let MaterialId::Light(_light_id) = hit.material {}
                    }

                    let material = world.get_material(hit.material);

                    vertex.outer_medium_id = material.outer_medium_id(hit.uv);
                    vertex.inner_medium_id = material.inner_medium_id(hit.uv);

                    // consider accumulating emission in some other form for trace_type == TransportMode::Importance situations, as mentioned in veach.
                    let maybe_wo: Option<Vec3> = material.generate(
                        hit.lambda,
                        hit.uv,
                        hit.transport_mode,
                        sampler.draw_2d(),
                        wi,
                    );

                    // what to do in this situation, where there is a wo and there's also emission?
                    let multi_emission = HeroEnergy(f32x4::new(
                        material
                            .emission(lambda.extract(0), hit.uv, hit.transport_mode, wi)
                            .0,
                        material
                            .emission(lambda.extract(1), hit.uv, hit.transport_mode, wi)
                            .0,
                        material
                            .emission(lambda.extract(2), hit.uv, hit.transport_mode, wi)
                            .0,
                        material
                            .emission(lambda.extract(3), hit.uv, hit.transport_mode, wi)
                            .0,
                    ));

                    // wo is generated in tangent space.

                    if let Some(wo) = maybe_wo {
                        // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                        let (multi_f, multi_pdf) = {
                            let (f0, pdf0) = material.bsdf(
                                lambda.extract(0),
                                hit.uv,
                                hit.transport_mode,
                                wi,
                                wo,
                            );
                            let (f1, pdf1) = material.bsdf(
                                lambda.extract(1),
                                hit.uv,
                                hit.transport_mode,
                                wi,
                                wo,
                            );
                            let (f2, pdf2) = material.bsdf(
                                lambda.extract(2),
                                hit.uv,
                                hit.transport_mode,
                                wi,
                                wo,
                            );
                            let (f3, pdf3) = material.bsdf(
                                lambda.extract(3),
                                hit.uv,
                                hit.transport_mode,
                                wi,
                                wo,
                            );
                            (
                                f32x4::new(f0.0, f1.0, f2.0, f3.0),
                                f32x4::new(pdf0.0, pdf1.0, pdf2.0, pdf3.0),
                            )
                        };
                        let (_reverse_multi_f, reverse_multi_pdf) = {
                            let (f0, pdf0) = material.bsdf(
                                lambda.extract(0),
                                hit.uv,
                                hit.transport_mode,
                                wo,
                                wi,
                            );
                            let (f1, pdf1) = material.bsdf(
                                lambda.extract(1),
                                hit.uv,
                                hit.transport_mode,
                                wo,
                                wi,
                            );
                            let (f2, pdf2) = material.bsdf(
                                lambda.extract(2),
                                hit.uv,
                                hit.transport_mode,
                                wo,
                                wi,
                            );
                            let (f3, pdf3) = material.bsdf(
                                lambda.extract(3),
                                hit.uv,
                                hit.transport_mode,
                                wo,
                                wi,
                            );
                            (
                                f32x4::new(f0.0, f1.0, f2.0, f3.0),
                                f32x4::new(pdf0.0, pdf1.0, pdf2.0, pdf3.0),
                            )
                        };
                        let cos_i = wo.z().abs();
                        let cos_o = wi.z().abs();
                        vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                        // if emission.0 > 0.0 {

                        // }

                        let hero_f = multi_f.extract(0);
                        let hero_pdf = multi_pdf.extract(0);
                        debug_assert!(hero_pdf >= 0.0, "pdf was less than 0 {:?}", hero_pdf);
                        if hero_pdf < 0.00000001 || hero_pdf.is_nan() {
                            break;
                        }
                        let rr_continue_prob = if bounce >= russian_roulette_start_index {
                            (hero_f / hero_pdf).min(1.0)
                        } else {
                            1.0
                        };
                        let russian_roulette_sample = sampler.draw_1d();
                        if russian_roulette_sample.x > rr_continue_prob {
                            break;
                        }
                        beta *= multi_f * cos_i.abs() / (rr_continue_prob * hero_pdf);
                        debug_assert!(
                            beta.is_finite().all(),
                            "{:?}, {:?}, {:?}, {:?}, ",
                            multi_f,
                            cos_i,
                            rr_continue_prob,
                            hero_pdf
                        );
                        vertex.pdf_forward = rr_continue_prob * multi_pdf / cos_i;

                        // consider handling delta distributions differently here, if deltas are ever added.
                        // eval pdf in reverse direction
                        vertex.pdf_backward = rr_continue_prob * reverse_multi_pdf / cos_o;

                        debug_assert!(
                            vertex.pdf_forward.extract(0) > 0.0 && vertex.pdf_forward.is_finite().all(),
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

                        vertices.push(HeroVertex::Surface(vertex));

                        // let beta_before_hit = beta;
                        // last_bsdf_pdf = pdf;

                        debug_assert!(
                            !beta.extract(0).is_nan(),
                            "{:?} {:?} {} {:?}",
                            beta,
                            multi_f,
                            cos_i,
                            multi_pdf
                        );

                        let outer = vertex.outer_medium_id;
                        let inner = vertex.inner_medium_id;
                        if wi.z() * wo.z() > 0.0 {
                            // scattering, so don't mess with volumes
                            // println!("reflect, {}, {}", outer, inner);
                        } else {
                            // transmitting, so remove appropriate medium from list and add new one. only applicable if inner != outer

                            if inner != outer {
                                // println!(
                                //     "transmit, {}, {}, {:?}, {:?}, {:?}",
                                //     outer, inner, wo, vertex.normal, tracked_mediums
                                // );
                                // print!("{:?} ", vertex.material_id);
                                if wo.z() < 0.0 {
                                    // println!("wo.z < 0, wi: {:?}, wo: {:?}", wi, wo);
                                    // transmitting from outer to inner. thus remove outer and add inner
                                    if outer != 0 {
                                        // only remove outer if it's not the Vacuum index.
                                        match tracked_mediums.iter().position(|e| *e == outer) {
                                            Some(index) => {
                                                tracked_mediums.remove(index);
                                            }
                                            None => {
                                                // println!(
                                                //     "warning: attempted to transition out of a medium that was not being tracked. tracked mediums already was {:?}. transmit from {} to {}, {:?}, {:?}.",
                                                //     tracked_mediums, outer, inner, wi, wo
                                                // );
                                            }
                                        }
                                    }
                                    if inner != 0 {
                                        // let insertion_index = tracked_mediums.binary_search(&inner);
                                        tracked_mediums.push(inner);
                                        tracked_mediums.sort_unstable();
                                    }
                                } else {
                                    // println!("wo.z > 0, wi: {:?}, wo: {:?}", wi, wo);
                                    // transmitting from inner to outer. thus remove inner and add outer, unless outer is vacuum.
                                    // also don't do anything if inner is vacuum.
                                    if inner != 0 {
                                        match tracked_mediums.iter().position(|e| *e == inner) {
                                            Some(index) => {
                                                tracked_mediums.remove(index);
                                            }
                                            None => {
                                                // println!(
                                                //     "warning: attempted to transition out of a medium that was not being tracked. tracked mediums already was {:?}. transmit from {} to {}, {:?}, {:?}.",
                                                //      tracked_mediums, inner,outer, wi, wo
                                                // );
                                            }
                                        }
                                    }
                                    if outer != 0 {
                                        tracked_mediums.push(outer);
                                        tracked_mediums.sort_unstable();
                                    }
                                }
                            }
                        }

                        // add normal to avoid self intersection
                        // also convert wo back to world space when spawning the new ray
                        ray = Ray::new(
                            hit.point
                                + hit.normal
                                    * NORMAL_OFFSET
                                    * if wo.z() > 0.0 { 1.0 } else { -1.0 },
                            frame.to_world(&wo).normalized(),
                        );
                    } else {
                        // hit a surface and didn't bounce.
                        if multi_emission.0.gt(f32x4::splat(0.0)).any() {
                            vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                            vertex.pdf_forward = f32x4::splat(0.0);
                            vertex.pdf_backward = f32x4::splat(1.0);
                            vertex.veach_g = veach_g(hit.point, wi.z().abs(), ray.origin, 1.0);
                            vertices.push(HeroVertex::Surface(vertex));
                        } else {
                            // this happens when the backside of a light is hit.
                        }
                        break;
                    }
                }
                HeroVertex::Medium(mut vertex) => {
                    let medium = &world.mediums[vertex.medium_id - 1];
                    let wi = -ray.direction;
                    let (wo, f_and_pdf) = medium.sample_p(
                        lambda.extract(0),
                        vertex.point.as_tuple(),
                        wi,
                        Sample2D::new_random_sample(),
                    );

                    // do russian roulette?

                    vertex.pdf_forward = f32x4::splat(f_and_pdf);
                    vertex.pdf_backward = f32x4::splat(f_and_pdf); // TODO: fix this, probably incorrect.
                    vertex.veach_g = veach_g(vertex.point, 1.0, ray.origin, 1.0);
                    vertices.push(HeroVertex::Medium(vertex));
                    // println!(
                    //     "medium interaction {}, wi = {:?}, wo = {:?}",
                    //     vertex.medium_id, wi, wo
                    // );
                    beta /= f_and_pdf; // pre divide by hero pdf
                    beta = beta.replace(0, beta.extract(0) * f_and_pdf);
                    debug_assert!(beta.is_finite().all(), "{:?} {:?}", beta, f_and_pdf);
                    for i in 1..4 {
                        let f_and_pdf =
                            medium.p(lambda.extract(i), vertex.point.as_tuple(), wi, wo);
                        beta = beta.replace(i, beta.extract(i) * f_and_pdf);
                        debug_assert!(beta.is_finite().all(), "{:?} {:?}", beta, f_and_pdf);
                    }

                    ray = Ray::new(vertex.point, wo);
                }
            }
        } else {
            // add a vertex when a camera ray hits the environment
            if trace_type == TransportMode::Importance {
                let ray_direction = ray.direction;
                let world_radius = world.get_world_radius();
                let at_env = ray_direction * world_radius;
                let vertex = HeroSurfaceVertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    ray.time,
                    lambda,
                    ray.direction,
                    Point3::from(at_env),
                    ray.direction,
                    (0.0, 0.0),
                    MaterialId::Light(0),
                    0,
                    HeroEnergy(beta),
                    f32x4::splat(0.0),
                    f32x4::splat(1.0 / (4.0 * PI)),
                    1.0,
                );
                debug_assert!(vertex.point.0.is_finite().all());
                // println!("sampling env and setting pdf_forward to 0");
                vertices.push(HeroVertex::Surface(vertex));
            }
            break;
        }
    }
    profile.bounce_rays += vertices.len();

    if additional_contribution.0.gt(f32x4::splat(0.0)).any() {
        Some(additional_contribution)
    } else {
        None
    }
}
