#![feature(portable_simd)]
extern crate rust_pathtracer as root;
#[macro_use]
extern crate tracing;

// std imports
use std::cmp::Ordering;
use std::fs::File;
use std::path::PathBuf;

// third party but non-subject-matter imports
#[cfg(feature = "preview")]
use minifb::WindowOptions;
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use structopt::StructOpt;

// our imports
use math::prelude::*;
use root::hittable::{HasBoundingBox, AABB};
use root::parsing::tonemap::TonemapSettings;
use root::parsing::{config::*, construct_world, get_settings, parse_tonemap_settings};
use root::prelude::*;
use root::renderer::output_film;
use root::world::{EnvironmentMap, Material, MaterialEnum};

// third party but subject-matter-relevant imports
use sdfu::ops::{HardMin, Union};
use sdfu::SDF;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use ultraviolet::vec::Vec3 as uvVec3;

trait Convert {
    fn convert(self) -> [f32; 4];
}

impl Convert for Point3 {
    fn convert(self) -> [f32; 4] {
        self.as_array()
    }
}

impl Convert for Vec3 {
    fn convert(self) -> [f32; 4] {
        self.as_array()
    }
}

#[inline(always)]
fn convert(p: impl Convert) -> uvVec3 {
    let [x, y, z, _] = p.convert();
    uvVec3::new(x, y, z)
}

#[inline(always)]
fn deconvert(v: uvVec3) -> f32x4 {
    f32x4::from_array([v.x, v.y, v.z, 0.0])
}

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long, default_value = "data/raymarch_config.toml")]
    pub config: String,
    #[structopt(short = "n", long)]
    pub dry_run: bool,
}
enum MarchResult {
    SurfaceIntersection {
        point: Point3,
        normal: Vec3,
        material: usize,
    },
    NoIntersection {
        // direction of ray in world space that didn't intersect anything
        direction: Vec3,
    },
}

// macro_rules! generate_primitive_enum {
//     ($($variant:ident => $type:ty )+) => {
//         #[derive(Copy, Clone)]
//         enum PrimitiveEnum {
//             $(
//                 $variant($type),
//             )+
//         }
//         impl SDF<f32, uvVec3> for PrimitiveEnum {
//             fn dist(&self, p: uvVec3) -> f32 {
//                 match self {
//                     $(
//                         PrimitiveEnum::$variant(inner) => {
//                             inner.dist(p)
//                         },
//                     )+
//                 }
//             }
//         }
//     };
// }

// generate_primitive_enum!(Sphere => Sphere<f32>);

trait MaterialTag {
    fn material(&self, _: uvVec3) -> usize {
        0
    }
}

#[derive(Copy, Clone)]
struct TaggedSDF<S>
where
    S: SDF<f32, uvVec3>,
{
    sdf: S,
    material: usize,
}

impl<S: SDF<f32, uvVec3>> TaggedSDF<S> {
    pub fn new(sdf: S, material: usize) -> Self {
        Self { sdf, material }
    }
}

impl<S: SDF<f32, uvVec3>> SDF<f32, uvVec3> for TaggedSDF<S> {
    fn dist(&self, p: uvVec3) -> f32 {
        self.sdf.dist(p)
    }
}

impl<S: SDF<f32, uvVec3>> MaterialTag for TaggedSDF<S> {
    fn material(&self, _: uvVec3) -> usize {
        self.material
    }
}

impl<S1: SDF<f32, uvVec3> + MaterialTag, S2: SDF<f32, uvVec3> + MaterialTag> MaterialTag
    for Union<f32, S1, S2, HardMin<f32>>
{
    fn material(&self, p: uvVec3) -> usize {
        let d0 = self.sdf1.dist(p);
        let d1 = self.sdf2.dist(p);
        match PartialOrd::partial_cmp(&d0, &d1) {
            Some(Ordering::Less) => self.sdf1.material(p),
            Some(Ordering::Greater) => self.sdf2.material(p),
            _ => {
                // just choose one?
                self.sdf1.material(p)
            }
        }
    }
}

#[derive(Copy, Clone)]
struct Mandlebulb {
    pub max_iterations: u32,
}

impl SDF<f32, uvVec3> for Mandlebulb {
    #[inline]
    fn dist(&self, p: uvVec3) -> f32 {
        let mut dz = 1.0;
        let mut last_p = uvVec3::new(0.0, 0.0, 0.0) + p;
        let mut mag = last_p.mag();
        for _ in 0..self.max_iterations {
            let mut distance = mag;
            let mut zenithal = (last_p.z / distance).acos();
            let mut azimuthal = last_p.y.atan2(last_p.x);

            // println!("{} {} {}", distance, zenithal, azimuthal);
            dz = 8.0 * mag.powi(7) * dz + 1.0;
            zenithal *= 8.0;
            azimuthal *= 8.0;
            distance = distance.powi(8);
            let (z_s, z_c) = zenithal.sin_cos();
            let (a_s, a_c) = azimuthal.sin_cos();
            last_p = uvVec3::new(distance * a_c * z_s, distance * a_s * z_s, distance * z_c) + p;
            mag = last_p.mag();
            if mag * mag > 256.0 {
                break;
            }
        }

        0.5 * mag.ln() * mag / dz
    }
}

struct Scene<S>
where
    S: SDF<f32, uvVec3> + MaterialTag,
{
    // primitives: Vec<Box<dyn SDF<f32, uvVec3>>>,
    // primitives: Vec<PrimitiveEnum>,
    primitives: S,
    environment: EnvironmentMap,
    materials: Vec<MaterialEnum>,
    // indexed by the same index as primitives, returns the material id for that primitive
    material_map: Vec<usize>,
    // max_depth: usize,
    world_aabb: AABB,
}

impl<S: SDF<f32, uvVec3> + MaterialTag> Scene<S> {
    fn sdf(&self, p: Point3) -> (f32, usize) {
        let p = convert(p);

        (self.primitives.dist(p), self.primitives.material(p))
    }
    fn normal(&self, p: Point3, threshold: f32) -> Vec3 {
        // get normal from prim[index]

        Vec3(deconvert(
            self.primitives.normals(threshold).normal_at(convert(p)),
        ))
    }

    pub fn march(
        &self,
        r: Ray,
        threshold: f32,
        offset: f32,
        flipped_sdf: bool,
        printout: bool,
    ) -> MarchResult {
        let mut distance = offset;
        let mut p = r.origin;
        let mut maybe_normal: Option<Vec3> = None;
        let mut maybe_material: Option<usize> = None;
        let max_depth = 100;
        for _ in 0..max_depth {
            p = r.point_at_parameter(distance);
            if !self.world_aabb.contains(p) {
                // already outside of world aabb
                if printout {
                    println!(
                        "aborting early, outside of world AABB. p = {:?}, total_time = {}",
                        p, distance
                    );
                }
                return MarchResult::NoIntersection {
                    direction: r.direction,
                };
            }
            let (mut current_distance, material_tag) = self.sdf(p);
            current_distance *= if flipped_sdf { -1.0 } else { 1.0 };
            if printout {
                println!(
                    "from point {:?}, time ({}) = {} + {} with nearest primitive = {}",
                    p,
                    distance + current_distance,
                    distance,
                    current_distance,
                    material_tag
                );
            }
            distance += current_distance;
            if current_distance.abs() < threshold {
                // reached boundary,
                // update p
                p = r.point_at_parameter(distance);
                maybe_normal = Some(self.normal(p, 0.01));
                maybe_material = Some(material_tag);
                break;
            }
        }
        match maybe_normal {
            Some(normal) => MarchResult::SurfaceIntersection {
                point: p,
                normal,
                material: self.material_map
                    [maybe_material.unwrap_or_else(|| panic!("strange error at point {:?}", p))],
            },
            None => MarchResult::NoIntersection {
                direction: r.direction,
            },
        }
    }
    pub fn color(
        &self,
        mut r: Ray,
        lambda: f32,
        bounces: u8,
        sampler: &mut Box<dyn Sampler>,
        printout: bool,
    ) -> XYZColor {
        let mut throughput = 1.0;
        let mut sum = 0.0;
        let mut flipped_sdf = false;
        // let mut last_bsdf_pdf = 0.0;
        for bounce in 0..bounces {
            match self.march(r, 0.001, 0.001, flipped_sdf, printout) {
                MarchResult::SurfaceIntersection {
                    point,
                    normal,
                    material,
                } => {
                    if printout {
                        println!("b{} = {:?}, {:?}, {}", bounce, point, normal, material);
                    }

                    assert!(normal.is_finite(), "{:?}", normal);
                    let frame = TangentFrame::from_normal(normal);
                    let wi = frame.to_local(&-r.direction);
                    let material = &self.materials[material];

                    let emission =
                        material.emission(lambda, UV(0.0, 0.0), TransportMode::Importance, wi);
                    if emission > 0.0 {
                        // do MIS based on last_bsdf_pdf
                        sum += throughput * emission * if true { wi.z().abs() } else { 1.0 };
                    }

                    let wo = material.generate(
                        lambda,
                        UV(0.0, 0.0),
                        TransportMode::Importance,
                        sampler.draw_2d(),
                        wi,
                    );
                    assert!(wo.map(|e| e.is_finite()).unwrap_or(true), "{:?}", wo);
                    if printout {
                        println!("wi {:?}, wo {:?}", wi, wo);
                    }
                    match wo {
                        Some(wo) => {
                            let (f, pdf) = material.bsdf(
                                lambda,
                                UV(0.0, 0.0),
                                TransportMode::Importance,
                                wi,
                                wo,
                            );
                            if printout {
                                println!("{:?} {:?}", f, pdf);
                            }
                            if pdf.is_nan() || *pdf == 0.0 {
                                break;
                            }
                            throughput *= wo.z().abs() * f / *pdf;
                            if throughput == 0.0 {
                                break;
                            }
                            r = Ray::new(
                                point + normal * NORMAL_OFFSET * wo.z().signum(),
                                frame.to_world(&wo).normalized(),
                            );
                            assert!(r.direction.is_finite(), "{:?}", r.direction);
                            if wo.z() * wi.z() < 0.0 {
                                // flipped inside of a material, thus invert the sdf until we do this again.
                                flipped_sdf = !flipped_sdf;
                            }
                        }
                        None => {
                            lazy_static! {
                                static ref LOGGED_CELL: std::sync::atomic::AtomicBool =
                                    std::sync::atomic::AtomicBool::new(false);
                            }
                            if !LOGGED_CELL.fetch_or(true, std::sync::atomic::Ordering::AcqRel) {
                                warn!("didn't bounce");
                            }
                            break;
                        }
                    }
                }
                MarchResult::NoIntersection { direction } => {
                    if printout {
                        println!("{} = {:?}", bounce, direction);
                    }
                    sum += throughput
                        * self
                            .environment
                            .emission(direction_to_uv(direction).into(), lambda);
                    break;
                }
            }
        }
        XYZColor::from(SingleWavelength::new(lambda, sum))
    }
}

macro_rules! find_and_add_material {
    ($world: expr, $materials:expr, $scrutinee: pat) => {
        $world
            .materials
            .iter()
            .enumerate()
            .find(|(_, material)| matches!(material, $scrutinee))
            .and_then(|(i, material)| {
                $materials.push(i);
                Some(material)
            })
            .unwrap();
    };
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(Level::TRACE)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    let opt = Opt::from_args();

    let settings = get_settings(opt.config).unwrap();

    let mut config = Config::from(settings);
    let render_settings = config.render_settings[0].clone();
    let (width, height) = (
        render_settings.resolution.width,
        render_settings.resolution.height,
    );

    let threads = config
        .render_settings
        .iter()
        .map(|i| &i.threads)
        .fold(1, |a, &b| a.max(b.unwrap_or(1)));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();
    // let camera = ProjectiveCamera::new(
    //     Point3::new(-5.0, 0.0, 0.0),
    //     Point3::ORIGIN,
    //     Vec3::Z,
    //     45.0,
    //     5.0,
    //     0.01,
    //     0.0,
    //     1.0,
    // )

    let scene_file_path = config.scene_file.clone();
    let mut handles = Vec::new();
    let world = construct_world(&mut config, PathBuf::from(scene_file_path), &mut handles).unwrap();

    let camera = world.cameras[0]
        .clone()
        .with_aspect_ratio(height as f32 / width as f32);
    // let mut primitives: Vec<PrimitiveEnum> = Vec::new();
    let mut material_map = Vec::new();

    // due to the culling optimization within construct world
    // assigning materials is tricky, currently.
    // you need to create dummy instances that reference materials so their data gets loaded
    // and you need to assign by finding that material through its type information here.
    // for more details, look at the definition of the find_material macro

    find_and_add_material!(world, material_map, MaterialEnum::Lambertian(_));
    find_and_add_material!(world, material_map, MaterialEnum::GGX(_));
    find_and_add_material!(world, material_map, MaterialEnum::DiffuseLight(_));
    // find_and_add_material!(world, material_map, MaterialEnum::SharpLight(_));

    let env_map = world.environment;

    let mut world_aabb = AABB::new(
        Point3::new(-10.0, -10.0, -10.0),
        Point3::new(10.0, 10.0, 10.0),
    );
    // world_aabb.grow_mut(&camera.origin);
    let (min, max) = {
        let aabb = camera.get_surface().unwrap().aabb();
        (aabb.min, aabb.max)
    };
    world_aabb.grow_mut(&min);
    world_aabb.grow_mut(&max);

    // let subject = TaggedSDF::new(
    //     sdfu::Sphere::new(1.0).union_smooth(
    //         sdfu::Sphere::new(1.0).translate(convert(Vec3::new(0.0, 0.0, 1.0))),
    //         0.3,
    //     ),
    //     1,
    // );

    let subject = TaggedSDF::new(Mandlebulb { max_iterations: 8 }.scale(1.0), 1);
    let scene_sdf = subject;

    println!("{}", subject.dist(uvVec3::new(1.0, 1.0, 1.0)));
    // return;

    // let local_light = TaggedSDF::new(
    //     sdfu::Sphere::new(1.0).translate(convert(Vec3::new(0.0, 2.0, 2.0))),
    //     2,
    // );
    // let scene_sdf = scene_sdf.union(local_light);

    let ground = TaggedSDF::new(
        sdfu::Box::new(convert(Vec3::new(10.0, 10.0, 0.1)))
            .translate(convert(Vec3::new(0.0, 0.0, -2.0))),
        0,
    );
    let scene_sdf = scene_sdf.union(ground);

    let scene = Scene {
        // primitives,
        primitives: scene_sdf,

        environment: env_map,
        materials: (*world.materials).clone(),
        material_map,
        // max_depth: 20,
        // world aabb needs to encompass camera
        world_aabb,
    };

    let wavelength_bounds = render_settings
        .wavelength_bounds
        .map(|e| e.into())
        .unwrap_or(BOUNDED_VISIBLE_RANGE);

    let mut render_film = Vec2D::new(width, height, XYZColor::BLACK);

    // let converter = Converter::sRGB;

    // let mut tonemapper = Clamp::new(-1.0, false, true);

    // force silence
    let tonemap_settings = match render_settings.tonemap_settings {
        TonemapSettings::Clamp {
            exposure,
            luminance_only,
            ..
        } => TonemapSettings::Clamp {
            exposure,
            luminance_only,
            silenced: true,
        },
        TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
            ..
        } => TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
            silenced: true,
        },
        TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
            ..
        } => TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
            silenced: true,
        },
    };
    let mut tonemapper = parse_tonemap_settings(tonemap_settings);

    let mut total_samples = 0;
    let samples_per_frame = 1;
    let bounces = render_settings.max_bounces.unwrap_or(10) as u8;

    if opt.dry_run {
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
        let (u, v) = (0.501, 0.5);
        let (r, _) = camera.get_ray(&mut sampler, lambda, u, v);

        let _ = scene.color(r, lambda, bounces, &mut sampler, true);
        return;
    }

    match config.renderer {
        RendererType::Naive => {
            total_samples = render_settings
                .max_samples
                .unwrap_or(render_settings.min_samples);
            let mut pb = ProgressBar::new((width * height * total_samples as usize) as u64);
            for _ in 0..total_samples {
                render_film
                    .buffer
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, pixel)| {
                        let (px, py) = (idx % width, idx / width);
                        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());

                        for _ in 0..samples_per_frame {
                            let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                            let film_sample = sampler.draw_2d();
                            let (u, v) = (
                                (px as f32 + film_sample.x) / width as f32,
                                (py as f32 + film_sample.y) / height as f32,
                            );
                            let (r, _) = camera.get_ray(&mut sampler, lambda, u, v);
                            *pixel += scene.color(r, lambda, bounces, &mut sampler, false);
                        }
                    });
                pb.add((width * height) as u64);
            }
            pb.finish();
        }
        RendererType::Tiled { .. } => {}
        #[cfg(feature = "preview")]
        RendererType::Preview { .. } => {
            root::window_loop(
                width,
                height,
                144,
                WindowOptions::default(),
                true,
                |_, window_buffer, width, height| {
                    render_film
                        .buffer
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(idx, pixel)| {
                            let (px, py) = (idx % width, idx / width);
                            let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());

                            for _ in 0..samples_per_frame {
                                let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
                                let film_sample = sampler.draw_2d();
                                let (u, v) = (
                                    (px as f32 + film_sample.x) / width as f32,
                                    (py as f32 + film_sample.y) / height as f32,
                                );
                                let (r, _) = camera.get_ray(&mut sampler, lambda, u, v);
                                *pixel += scene.color(r, lambda, bounces, &mut sampler, false);
                            }
                        });
                    total_samples += samples_per_frame;

                    let factor = 1.0 / (total_samples as f32 + 1.0);

                    update_window_buffer(window_buffer, &render_film, tonemapper.as_mut(), factor);
                },
            );
        }
    }
    output_film(
        &render_settings,
        &render_film,
        1.0 / (total_samples as f32 + 1.0),
    );
}
