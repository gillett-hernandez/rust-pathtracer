use crate::{mediums::MediumEnum, prelude::*};

use log_once::warn_once;

use crate::hittable::HitRecord;
use crate::mediums::Medium;
use crate::profile::Profile;
use crate::world::World;

use std::{
    ops::{Mul, MulAssign},
    sync::Arc,
};

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

// TODO: maybe instead of adding the ToScalar trait bound with f32, we could generalize HitRecord.
// TODO: actually, just refactor HitRecord completely.
#[derive(Debug, Copy, Clone)]
pub struct SurfaceVertex<L: Field, E: Field> {
    pub vertex_type: VertexType,
    pub time: f32,
    pub lambda: L,
    pub local_wi: Vec3,
    pub point: Point3,
    pub normal: Vec3,
    pub uv: (f32, f32),
    pub material_id: MaterialId,
    pub instance_id: InstanceId,
    pub throughput: E,
    pub pdf_forward: PDF<E, SolidAngle>,
    pub pdf_backward: PDF<E, SolidAngle>,
    pub veach_g: f32,
    pub inner_medium_id: u8,
    pub outer_medium_id: u8,
}

impl<L: Field, E: Field> SurfaceVertex<L, E> {
    pub fn new(
        vertex_type: VertexType,
        time: f32,
        lambda: L,
        local_wi: Vec3,
        point: Point3,
        normal: Vec3,
        uv: (f32, f32),
        material_id: MaterialId,
        instance_id: InstanceId,
        throughput: E,
        pdf_forward: PDF<E, SolidAngle>,
        pdf_backward: PDF<E, SolidAngle>,
        veach_g: f32,
        inner_medium_id: u8,
        outer_medium_id: u8,
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
            inner_medium_id,
            outer_medium_id,
        }
    }

    pub fn default() -> Self {
        SurfaceVertex::new(
            VertexType::Eye,
            0.0,
            L::ZERO,
            Vec3::ZERO,
            Point3::ORIGIN,
            Vec3::ZERO,
            (0.0, 0.0),
            MaterialId::Material(0),
            0,
            E::ZERO,
            E::ZERO.into(),
            E::ZERO.into(),
            0.0,
            0,
            0,
        )
    }
}

impl<L: Field + ToScalar<f32>, E: Field> From<SurfaceVertex<L, E>> for HitRecord {
    fn from(data: SurfaceVertex<L, E>) -> Self {
        let transport_mode = match data.vertex_type {
            VertexType::Light | VertexType::LightSource(_) => TransportMode::Radiance,
            VertexType::Camera | VertexType::Eye => TransportMode::Importance,
        };
        HitRecord::new(
            data.time,
            data.point,
            data.uv,
            data.lambda.to_scalar(),
            data.normal,
            data.material_id,
            data.instance_id,
            Some(transport_mode),
        )
    }
}

pub fn veach_v(world: &Arc<World>, point0: Point3, point1: Point3) -> bool {
    // returns whether the points are mutually visible.

    let diff = point1 - point0;
    let norm = diff.norm();
    let tmax = norm * 0.99;
    let point0_to_point1 = Ray::new_with_time_and_tmax(point0, diff / norm, 0.0, tmax);
    let hit = world.hit(point0_to_point1, INTERSECTION_TIME_OFFSET, tmax);

    hit.is_none()
}

pub fn veach_g(point0: Point3, cos_i: f32, point1: Point3, cos_o: f32) -> f32 {
    (cos_i * cos_o).abs() / (point1 - point0).norm_squared()
}

pub fn random_walk<L, E>(
    mut ray: Ray,
    lambda: L,
    bounce_limit: u16,
    start_throughput: E,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<SurfaceVertex<L, E>>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
    ignore_backward: bool,
) where
    MaterialEnum: Material<L, E>,
    L: Field + ToScalar<f32>,
    E: Field + ToScalar<f32> + FromScalar<f32> + Mul<f32, Output = E>,
{
    let mut beta = start_throughput;
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.0, ray.tmax) {
            hit.lambda = lambda.to_scalar();
            hit.transport_mode = trace_type;

            let frame = TangentFrame::from_normal(hit.normal);
            let wi = frame.to_local(&-ray.direction).normalized();
            let mut vertex = SurfaceVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                wi,
                hit.point,
                hit.normal,
                hit.uv,
                hit.material,
                hit.instance_id,
                beta,
                PDF::new(E::ONE),
                PDF::new(E::ONE),
                1.0,
                0,
                0,
            );

            let material = world.get_material(hit.material);
            let emission =
                Material::<L, E>::emission(material, lambda, hit.uv, hit.transport_mode, wi);
            match (hit.material, trace_type) {
                (MaterialId::Camera(_camera_id), TransportMode::Radiance) => {
                    vertex.vertex_type = VertexType::Camera;
                    vertices.push(vertex);
                    break;
                }
                // if directly hit a light while tracing a camera path.
                (MaterialId::Light(_light_id), TransportMode::Importance) => {
                    vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                }

                // TODO: think about sampling lights when doing LT. theoretically it should be possible and not unphysical
                _ => {}
            }

            // consider accumulating emission in some other form for trace_type == TransportMode::Importance situations, as mentioned in veach.
            let maybe_wo: Option<Vec3> = Material::<L, E>::generate(
                material,
                lambda,
                hit.uv,
                hit.transport_mode,
                sampler.draw_2d(),
                wi,
            );

            // wo is generated in tangent space.

            if let Some(wo) = maybe_wo {
                // Material::bsdf(&material, hit, lambda, uv, transport_mode, wi, wo)
                let (f, pdf) =
                    Material::<L, E>::bsdf(material, lambda, hit.uv, hit.transport_mode, wi, wo);
                let cos_o = wo.z().abs();
                let cos_i = wi.z().abs();
                vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);
                if !vertex.veach_g.is_finite() {
                    warn_once!(
                        "veach g was inf, {:?} {:?} {} {}",
                        hit.point,
                        ray.origin,
                        cos_i,
                        cos_o
                    );
                }

                debug_assert!(
                    matches!(
                        MyPartialOrd::partial_cmp(&*pdf, &E::ZERO),
                        Some(std::cmp::Ordering::Greater)
                    ),
                    "pdf was less than 0 {:?}",
                    pdf
                );
                if pdf.check_nan().coerce(true) {
                    break;
                }

                // TODO: confirm whether russian roulette is solely based on f/pdf or if it can take into account beta.
                let rr_continue_prob: PDF<f32, Uniform01> =
                    if bounce >= russian_roulette_start_index {
                        // f/pdf % probability of continuing, i.e. high throughput = high chance of continuing
                        (f.to_scalar() / (*pdf).to_scalar()).min(1.0).into()
                    } else {
                        // 100% probability of continuing
                        1.0.into()
                    };

                // consider handling delta distributions differently here, if deltas are ever added.
                // dividing by cos_o seems to imply that this pdf is a SolidAngle pdf
                //
                vertex.pdf_forward = pdf * (*rr_continue_prob / cos_o);

                // eval pdf in reverse direction
                // FIXME: confirm that transport mode doesn't need to be flipped

                if !ignore_backward {
                    vertex.pdf_backward = Material::<L, E>::bsdf(
                        material,
                        lambda,
                        hit.uv,
                        hit.transport_mode,
                        wo,
                        wi,
                    )
                    .1 * (*rr_continue_prob / cos_i);
                }
                // last_vertex = Some(vertex);
                vertices.push(vertex);

                beta *= f / *vertex.pdf_forward;
                // figure out a better way to express this condition, since it might not apply in this exact way if E is f32x4
                if *vertex.pdf_forward == E::ZERO {
                    beta = E::ZERO;
                }
                debug_assert!(
                    !beta.check_nan().coerce(true),
                    "{:?} {:?} {} {:?}",
                    beta,
                    f,
                    cos_o,
                    pdf
                );
                if beta == E::ZERO {
                    break;
                }

                let russian_roulette_sample = sampler.draw_1d();
                if russian_roulette_sample.x > *rr_continue_prob {
                    break;
                }

                // add normal to avoid self intersection
                // also convert wo back to world space when spawning the new ray
                ray = Ray::new(
                    hit.point + hit.normal * NORMAL_OFFSET * wo.z().signum(),
                    frame.to_world(&wo).normalized(),
                );
            } else {
                // hit a surface and didn't bounce.

                // will correctly identify as Greater even if only one lane is greater than 0, as of rust_cg_math#879d90b5
                // TODO: tag that revision on git and pin it in Cargo.toml
                if matches!(emission.partial_cmp(&E::ZERO), Some(Ordering::Greater)) {
                    vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                    vertex.pdf_forward = E::ZERO.into();
                    vertex.pdf_backward = E::ONE.into();
                    vertex.veach_g = veach_g(hit.point, wi.z().abs(), ray.origin, 1.0);
                    vertices.push(vertex);
                }
                break;
            }
        } else {
            // add a vertex when a camera ray hits the environment
            // TODO: maybe resample the environment here?
            if trace_type == TransportMode::Importance {
                let world_radius = world.radius;
                let at_env = ray.direction * world_radius;
                let vertex = SurfaceVertex::new(
                    VertexType::LightSource(LightSourceType::Environment),
                    ray.time,
                    lambda,
                    Vec3::Z,
                    Point3::from(at_env),
                    ray.direction,
                    // delay computing uv
                    (0.0, 0.0),
                    MaterialId::Light(0),
                    0,
                    beta,
                    E::ZERO.into(),
                    E::from_scalar((4.0 * PI).recip()).into(),
                    1.0,
                    0,
                    0,
                );
                debug_assert!(vertex.point.0.is_finite().all());
                // println!("sampling env and setting pdf_forward to 0");
                vertices.push(vertex);
            }
            break;
        }
    }
    profile.bounce_rays += vertices.len();
}
/*
pub fn random_walk_hero(
    mut ray: Ray,
    lambda: f32x4,
    bounce_limit: u16,
    start_throughput: f32x4,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<SurfaceVertex>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
) {
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.0, ray.tmax) {
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
                if let MaterialId::Light(_light_id) = hit.material {
                    // TODO: handle this
                }
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
                let world_radius = world.radius;
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
} */

#[derive(Debug, Copy, Clone)]
pub struct MediumVertex<L: Field, E: Field> {
    pub vertex_type: VertexType,
    pub time: f32,
    pub lambda: L,
    pub wi: Vec3,
    pub point: Point3,
    pub uvw: (f32, f32, f32),
    pub medium_id: MediumId,
    pub instance_id: InstanceId,
    pub throughput: E,
    pub pdf_forward: PDF<E, SolidAngle>,
    pub pdf_backward: PDF<E, SolidAngle>,
    pub veach_g: f32,
}

impl<L: Field, E: Field> MediumVertex<L, E> {
    pub fn new(
        vertex_type: VertexType,
        time: f32,
        lambda: L,
        wi: Vec3,
        point: Point3,
        uvw: (f32, f32, f32),
        medium_id: MediumId,
        instance_id: InstanceId,
        throughput: E,
        pdf_forward: PDF<E, SolidAngle>,
        pdf_backward: PDF<E, SolidAngle>,
        veach_g: f32,
    ) -> Self {
        MediumVertex {
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
        MediumVertex::new(
            VertexType::Eye,
            0.0,
            L::ZERO,
            Vec3::ZERO,
            Point3::ORIGIN,
            (0.0, 0.0, 0.0),
            0,
            0,
            E::ZERO,
            PDF::new(E::ZERO),
            PDF::new(E::ZERO),
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
pub enum Vertex<L: Field + ToScalar<f32>, E: Field> {
    Surface(SurfaceVertex<L, E>),
    Medium(MediumVertex<L, E>),
}

impl<L: Field + ToScalar<f32>, E: Field> Vertex<L, E> {
    pub fn point(&self) -> Point3 {
        match self {
            Vertex::Medium(v) => v.point,
            Vertex::Surface(v) => v.point,
        }
    }
    // pub fn pdf_forward<S: Scalar>(&self, cos_theta: S) -> PDF<E, Throughput>
    // where
    //     E: ToScalar<S>,
    // {
    //     match self {
    //         Vertex::Medium(v) => v.pdf_forward.convert_to_projected_solid_angle(cos_theta),
    //         Vertex::Surface(v) => v.pdf_forward,
    //     }
    // }
    pub fn cos(&self, vec: Vec3) -> f32 {
        match self {
            Vertex::Medium(_v) => 1.0,
            Vertex::Surface(v) => v.normal * vec,
        }
    }
}

pub fn random_walk_medium<L, E>(
    mut ray: Ray,
    lambda: L,
    bounce_limit: u16,
    start_throughput: E,
    trace_type: TransportMode,
    sampler: &mut Box<dyn Sampler>,
    world: &Arc<World>,
    vertices: &mut Vec<Vertex<L, E>>,
    russian_roulette_start_index: u16,
    profile: &mut Profile,
) where
    L: Field + ToScalar<f32>,
    E: Field + ToScalar<f32> + FromScalar<f32> + Mul<f32, Output = E> + MulAssign<f32>,
    MaterialEnum: Material<L, E>,
    MediumEnum: Medium<L, E>,
{
    let mut beta = start_throughput;
    // let mut last_bsdf_pdf = PDF::from(0.0);
    let mut tracked_mediums: Vec<MediumId> = Vec::new();
    for bounce in 0..bounce_limit {
        if let Some(mut hit) = world.hit(ray, 0.0, ray.tmax) {
            hit.lambda = lambda.to_scalar();
            hit.transport_mode = trace_type;
            let mut surface_vertex = SurfaceVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                -ray.direction,
                hit.point,
                hit.normal,
                hit.uv,
                hit.material,
                hit.instance_id,
                beta,
                E::ONE.into(),
                E::ONE.into(),
                1.0,
                0,
                0,
            );

            let mut medium_vertex = MediumVertex::new(
                trace_type.into(),
                hit.time,
                lambda,
                -ray.direction,
                hit.point,
                (0.0, 0.0, 0.0),
                0,
                hit.instance_id,
                beta,
                E::ONE.into(),
                E::ONE.into(),
                1.0,
            );

            let mut vertex = Vertex::Surface(surface_vertex);

            let mut hero_weight = E::ONE;
            let mut hero_tr = E::ONE;
            for medium_id in tracked_mediums.iter() {
                debug_assert!(*medium_id > 0);
                let medium = &world.mediums[(*medium_id - 1) as usize];
                let (p, tr, scatter) =
                    medium.sample(lambda.to_scalar(), ray, Sample1D::new_random_sample());
                if scatter {
                    let t = (p - ray.origin).norm();
                    if t < medium_vertex.time {
                        medium_vertex.time = t;
                        medium_vertex.point = p;
                        hero_weight = tr;
                        // hero_tr = medium.tr(ToScalar::convert(0), ray.origin, p);
                        medium_vertex.medium_id = *medium_id;
                        // println!(
                        //     "overrode surface vertex with medium vertex, p = {:?}",
                        //     medium_vertex.point
                        // );
                        vertex = Vertex::Medium(medium_vertex);
                    }
                }
            }
            // multiply in hero weight, since it includes some of the hero pdf information and that would be lost if unaccounted for.
            // hero weight also includes tr.
            beta *= hero_weight;
            let mut combined_throughput = E::ONE;
            for medium_id in tracked_mediums.iter() {
                debug_assert!(*medium_id > 0);
                let medium = &world.mediums[(*medium_id - 1) as usize];
                combined_throughput *=
                    Medium::<L, E>::tr(medium, lambda, ray.origin, medium_vertex.point);
            }

            // divide out hero_tr for all wavelengths, since it was included in overall beta mult.
            beta *= combined_throughput / hero_tr;

            // multiply hero_tr back in for only hero wavelength.
            // beta = beta.replace(0, beta.extract(0) * hero_tr);

            match vertex {
                Vertex::Surface(mut vertex) => {
                    let frame = TangentFrame::from_normal(hit.normal);
                    let wi = frame.to_local(&-ray.direction).normalized();

                    if matches!(hit.material, MaterialId::Camera(_)) {
                        if trace_type == TransportMode::Radiance {
                            // if hit camera directly while tracing a light path
                            surface_vertex.vertex_type = VertexType::Camera;
                            vertices.push(Vertex::Surface(surface_vertex));
                        }
                        break;
                    } else if matches!(hit.material, MaterialId::Light(_)) {
                        // if directly hit a light while tracing a camera path.
                        // TODO: handle this
                    }

                    let material = world.get_material(hit.material);

                    vertex.outer_medium_id = material.outer_medium_id(hit.uv);
                    vertex.inner_medium_id = material.inner_medium_id(hit.uv);

                    // consider accumulating emission in some other form for trace_type == TransportMode::Importance situations, as mentioned in veach.
                    let maybe_wo: Option<Vec3> = Material::<L, E>::generate(
                        material,
                        lambda,
                        hit.uv,
                        hit.transport_mode,
                        sampler.draw_2d(),
                        wi,
                    );

                    // what to do in this situation, where there is a wo and there's also emission?
                    let emission = Material::<L, E>::emission(
                        material,
                        lambda,
                        hit.uv,
                        hit.transport_mode,
                        wi,
                    );

                    // wo is generated in tangent space.

                    if let Some(wo) = maybe_wo {
                        // NOTE! cos_i and cos_o seem to have somewhat reversed names.
                        let (f, pdf) = Material::<L, E>::bsdf(
                            material,
                            lambda,
                            hit.uv,
                            hit.transport_mode,
                            wi,
                            wo,
                        );
                        let (reverse_f, reverse_pdf) = Material::<L, E>::bsdf(
                            material,
                            lambda,
                            hit.uv,
                            hit.transport_mode,
                            wo,
                            wi,
                        );
                        let cos_i = wo.z().abs();
                        let cos_o = wi.z().abs();
                        vertex.veach_g = veach_g(hit.point, cos_i, ray.origin, cos_o);

                        let hero_f = f.to_scalar();
                        let hero_pdf: PDF<_, SolidAngle> = pdf.to_scalar().into();
                        debug_assert!(*hero_pdf >= 0.0, "pdf was less than 0 {:?}", hero_pdf);
                        if *hero_pdf == 0.0 || hero_pdf.is_nan() {
                            break;
                        }
                        let rr_continue_prob: PDF<f32, Uniform01> =
                            if bounce >= russian_roulette_start_index {
                                (hero_f / *hero_pdf).min(1.0).into()
                            } else {
                                1.0.into()
                            };
                        let russian_roulette_sample = sampler.draw_1d();
                        if russian_roulette_sample.x > *rr_continue_prob {
                            break;
                        }
                        beta *= f * cos_i.abs() * (*rr_continue_prob * *hero_pdf).recip();
                        debug_assert!(
                            // all not inf
                            !beta.check_inf().coerce(true),
                            "{:?}, {:?}, {:?}, {:?}, ",
                            f,
                            cos_i,
                            rr_continue_prob,
                            hero_pdf
                        );
                        vertex.pdf_forward = pdf * (*rr_continue_prob / cos_i);

                        vertex.pdf_backward = reverse_pdf * (*rr_continue_prob / cos_o);

                        debug_assert!(
                            vertex.pdf_forward.to_scalar() > 0.0 && !vertex.pdf_forward.check_inf().coerce(true),
                            "pdf forward was 0 for material {:?} at vertex {:?}. wi: {:?}, wo: {:?}, cos_o: {}, cos_i: {}, rrcont={:?}",
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

                        vertices.push(Vertex::Surface(vertex));

                        // let beta_before_hit = beta;
                        // last_bsdf_pdf = pdf;

                        debug_assert!(
                            !beta.check_nan().coerce(true),
                            "{:?} {:?} {} {:?}",
                            beta,
                            f,
                            cos_i,
                            pdf
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
                                        match tracked_mediums
                                            .iter()
                                            .position(|e| (*e as u8) == outer)
                                        {
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
                                        match tracked_mediums
                                            .iter()
                                            .position(|e| (*e as u8) == inner)
                                        {
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
                        if matches!(emission.partial_cmp(&E::ZERO), Some(Ordering::Greater)) {
                            // if emission.0.gt(f32x4::splat(0.0)).any() {
                            vertex.vertex_type = VertexType::LightSource(LightSourceType::Instance);
                            vertex.pdf_forward = E::ZERO.into();
                            vertex.pdf_backward = E::ONE.into();
                            vertex.veach_g = veach_g(hit.point, wi.z().abs(), ray.origin, 1.0);
                            vertices.push(Vertex::Surface(vertex));
                        } else {
                            // this happens when the backside of a light is hit.
                        }
                        break;
                    }
                }
                Vertex::Medium(mut vertex) => {
                    debug_assert!(vertex.medium_id > 0);
                    let medium = &world.mediums[(vertex.medium_id - 1) as usize];
                    let wi = -ray.direction;
                    let [u, v, w, _] = vertex.point.as_array();
                    let (wo, phase) = medium.sample_p(
                        lambda.to_scalar(),
                        (u, v, w),
                        wi,
                        Sample2D::new_random_sample(),
                    );

                    // let phase = Medium::<L, E>::p(medium, lambda, (u, v, w), wi, wo);

                    // phase from sample_p is a pdf (satisfies normalization condition)
                    // but currently, phase from ::p does not satisfy the normalization condition.
                    // TODO: figure this out, maybe have p return two values, one being the unnormalized phase and the other being the normalized phase
                    // do russian roulette?

                    vertex.pdf_forward = PDF::new(E::from_scalar(*phase));
                    vertex.pdf_backward = PDF::new(E::from_scalar(*phase)); // phase functions are reciprocal
                    vertex.veach_g = veach_g(vertex.point, 1.0, ray.origin, 1.0);
                    vertices.push(Vertex::Medium(vertex));
                    // println!(
                    //     "medium interaction {}, wi = {:?}, wo = {:?}",
                    //     vertex.medium_id, wi, wo
                    // );
                    // beta /= phase.0; // pre divide by hero pdf
                    // beta = beta.replace(0, beta.extract(0) * phase.0);
                    // debug_assert!(beta.is_finite().all(), "{:?} {:?}", beta, phase);

                    // let [u, v, w, _] = vertex.point.as_array();
                    // for i in 1..4 {
                    //     let f_and_pdf = medium.p(lambda.extract(i), (u, v, w), wi, wo);
                    //     beta = beta.replace(i, beta.extract(i) * f_and_pdf);
                    //     debug_assert!(beta.is_finite().all(), "{:?} {:?}", beta, f_and_pdf);
                    // }

                    ray = Ray::new(vertex.point, wo);
                }
            }
        } else {
            // add a vertex when a camera ray hits the environment
            if trace_type == TransportMode::Importance {
                let ray_direction = ray.direction;
                let world_radius = world.radius;
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
                    E::ZERO.into(),
                    E::from_scalar(1.0 / (4.0 * PI)).into(),
                    1.0,
                    0,
                    0,
                );
                debug_assert!(vertex.point.0.is_finite().all());
                // println!("sampling env and setting pdf_forward to 0");
                vertices.push(Vertex::Surface(vertex));
            }
            break;
        }
    }
    profile.bounce_rays += vertices.len();
}
