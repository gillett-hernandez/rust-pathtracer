#[macro_use]
extern crate log;
extern crate rust_pathtracer as root;
extern crate simplelog;

use std::cmp::Ordering;
use std::fs::File;
use std::ops::{Add, Div, Mul, Neg, RangeInclusive, Sub};

use log_once::warn_once;
use math::curves::*;
use math::prelude::{direction_to_uv, Point3, XYZColor};
use math::ray::Ray;
use math::sample::{RandomSampler, Sampler};
use math::spectral::{SingleWavelength, WavelengthEnergyTrait, BOUNDED_VISIBLE_RANGE};
use math::*;

use math::tangent_frame::TangentFrame;
use math::vec::Vec3;
use packed_simd::f32x4;
use pbr::ProgressBar;
use root::camera::ProjectiveCamera;
use root::hittable::{HasBoundingBox, AABB};
use root::parsing::tonemap::TonemapSettings;
use root::parsing::{
    config::*, construct_world, get_settings, load_scene, parse_config_and_cameras,
    parse_tonemapper,
};
use root::prelude::Camera;
use root::renderer::{output_film, Film};
use root::rgb_to_u32;
use root::tonemap::{Clamp, Converter, Tonemapper};
use root::world::{EnvironmentMap, Material, MaterialEnum};
use root::*;

use log::LevelFilter;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use sdfu::ops::{HardMin, Union};
use simplelog::{ColorChoice, CombinedLogger, TermLogger, TerminalMode, WriteLogger};
use structopt::StructOpt;

use sdfu::{Sphere, SDF};
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
    f32x4::new(v.x, v.y, v.z, 0.0)
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
    fn material(&self, p: uvVec3) -> usize {
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
    fn material(&self, p: uvVec3) -> usize {
        self.material
    }
}

impl<S1: SDF<f32, uvVec3> + MaterialTag, S2: SDF<f32, uvVec3> + MaterialTag> MaterialTag
    for Union<f32, S1, S2, HardMin<f32>>
{
    fn material(&self, p: uvVec3) -> usize {
        // let minfunc = self.min_func;
        // minfunc.
        let d0 = self.sdf1.dist(p);
        let d1 = self.sdf2.dist(p);
        match d0.partial_cmp(&d1) {
            Some(Ordering::Less) => self.sdf1.material(p),
            Some(Ordering::Greater) => self.sdf2.material(p),
            Some(Ordering::Equal) => {
                // just choose one?
                self.sdf1.material(p)
            }
            None => unreachable!(),
        }
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
        let mut min_time = f32::INFINITY;
        let mut selected = 0;

        // for (index, prim) in self.primitives.iter().enumerate() {
        //     let time = prim.dist(convert(p));

        //     if time < min_time {
        //         min_time = time;
        //         selected = index
        //     }
        // }
        selected = self.primitives.material(convert(p));
        min_time = self.primitives.dist(convert(p));
        (min_time, selected)
    }
    fn normal(&self, p: Point3, threshold: f32) -> Vec3 {
        // get normal from prim[index]
        // Vec3(deconvert(
        //     self.primitives[index]
        //         .normals(threshold)
        //         .normal_at(convert(p)),
        // ))
        Vec3(deconvert(
            self.primitives.normals(threshold).normal_at(convert(p)),
        ))
    }
    // fn material(&self, p: Point3, threshold: f32) -> Option<usize> {
    //     // dead code?
    //     for (i, prim) in self.primitives.iter().enumerate() {
    //         if prim.dist(convert(p)) < threshold {
    //             // considered on or inside surface
    //             return Some(i);
    //         }
    //     }
    //     None
    // }
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
        let mut last_bsdf_pdf = 0.0;
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
                    let frame = TangentFrame::from_normal(normal);
                    let wi = frame.to_local(&-r.direction);
                    let material = &self.materials[material];

                    let emission =
                        material.emission(lambda, (0.0, 0.0), TransportMode::Importance, wi);
                    if emission > 0.0 {
                        // do MIS based on last_bsdf_pdf
                        sum += throughput * emission * if true { wi.z().abs() } else { 1.0 };
                    }

                    let wo = material.generate(
                        lambda,
                        (0.0, 0.0),
                        TransportMode::Importance,
                        sampler.draw_2d(),
                        wi,
                    );
                    if printout {
                        println!("wi {:?}, wo {:?}", wi, wo);
                    }
                    match wo {
                        Some(wo) => {
                            let (f, pdf) = material.bsdf(
                                lambda,
                                (0.0, 0.0),
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
                            if wo.z() * wi.z() < 0.0 {
                                // flipped inside of a material, thus invert the sdf until we do this again.
                                flipped_sdf = !flipped_sdf;
                            }
                        }
                        None => {
                            warn_once!("didn't bounce");
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
                            .emission(direction_to_uv(direction), lambda);
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
            .find(|(index, material)| matches!(material, $scrutinee))
            .and_then(|(i, material)| {
                $materials.push(i);
                Some(material)
            })
            .unwrap();
    };
}

fn main() {
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Warn,
            simplelog::Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            simplelog::Config::default(),
            File::create("main.log").unwrap(),
        ),
    ])
    .unwrap();

    let opt = Opt::from_args();

    let settings = get_settings(opt.config).unwrap();
    let (config, cameras) = parse_config_and_cameras(settings);
    let render_settings = &config.render_settings[0];
    let (width, height) = (
        render_settings.resolution.width,
        render_settings.resolution.height,
    );

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
    let camera = cameras[0]
        .clone()
        .with_aspect_ratio(height as f32 / width as f32);

    let world = construct_world(config.scene_file).unwrap();

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

    let subject = TaggedSDF::new(
        sdfu::Sphere::new(1.0).union_smooth(
            sdfu::Sphere::new(1.0).translate(convert(Vec3::new(0.0, 0.0, 1.0))),
            0.3,
        ),
        1,
    );

    // let subject = TaggedSDF::new(
    //     sdfu::,
    //     1,
    // );
    let local_light = TaggedSDF::new(
        sdfu::Sphere::new(1.0).translate(convert(Vec3::new(0.0, 2.0, 2.0))),
        2,
    );

    let ground = TaggedSDF::new(
        sdfu::Box::new(convert(Vec3::new(10.0, 10.0, 0.1)))
            .translate(convert(Vec3::new(0.0, 0.0, -2.0))),
        0,
    );

    let scene_sdf = subject.union(local_light).union(ground);

    let scene = Scene {
        // primitives,
        primitives: scene_sdf,

        environment: env_map,
        materials: world.materials,
        material_map,
        // max_depth: 20,
        // world aabb needs to encompass camera
        world_aabb,
    };

    let wavelength_bounds = render_settings
        .wavelength_bounds
        .map(|e| e.into())
        .unwrap_or(BOUNDED_VISIBLE_RANGE);

    let mut render_film = Film::new(width, height, XYZColor::BLACK);

    // let converter = Converter::sRGB;

    // let mut tonemapper = Clamp::new(-1.0, false, true);

    // force silence
    let tonemap_settings = match render_settings.tonemap_settings {
        TonemapSettings::Clamp {
            exposure,
            luminance_only,
            silenced,
        } => TonemapSettings::Clamp {
            exposure,
            luminance_only,
            silenced: true,
        },
        TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
            silenced,
        } => TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
            silenced: true,
        },
        TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
            silenced,
        } => TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
            silenced: true,
        },
    };
    let (mut tonemapper, converter) = parse_tonemapper(tonemap_settings);

    let mut total_samples = 0;
    let samples_per_frame = 1;
    let bounces = render_settings.max_bounces.unwrap_or(10) as u8;

    if opt.dry_run {
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
        let (u, v) = (0.501, 0.5);
        let (r, pdf) = camera.get_ray(&mut sampler, lambda, u, v);

        let color = scene.color(r, lambda, bounces, &mut sampler, true);
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
                            let (r, pdf) = camera.get_ray(&mut sampler, lambda, u, v);
                            *pixel += scene.color(r, lambda, bounces, &mut sampler, false);
                        }
                    });
                pb.add((width * height) as u64);
            }
            pb.finish();
        }
        RendererType::Tiled { tile_size } => {}
        RendererType::Preview { .. } => {
            root::window_loop(
                width,
                height,
                144,
                WindowOptions::default(),
                true,
                |_, film, width, height| {
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
                                let (r, pdf) = camera.get_ray(&mut sampler, lambda, u, v);
                                *pixel += scene.color(r, lambda, bounces, &mut sampler, false);
                            }
                        });
                    total_samples += samples_per_frame;

                    tonemapper.initialize(&render_film, 1.0 / (total_samples as f32 + 1.0));
                    film.par_iter_mut().enumerate().for_each(|(pixel_idx, v)| {
                        let y: usize = pixel_idx / width;
                        let x: usize = pixel_idx % width;
                        let [r, g, b, _]: [f32; 4] = converter
                            .transfer_function(
                                tonemapper.map(&render_film, (x as usize, y as usize)),
                                false,
                            )
                            .into();
                        *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                    });
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
