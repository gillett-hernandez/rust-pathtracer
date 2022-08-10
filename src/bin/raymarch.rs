#[macro_use]
extern crate log;
extern crate rust_pathtracer as root;
extern crate simplelog;

use std::fs::File;
use std::ops::{Add, Div, Mul, Neg, RangeInclusive, Sub};

use log_once::warn_once;
use math::curves::*;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use math::*;

use root::camera::ProjectiveCamera;
use root::hittable::{HasBoundingBox, AABB};
use root::parsing::{config::*, construct_world, load_scene, parse_tonemapper};
use root::renderer::Film;
use root::rgb_to_u32;
use root::tonemap::{Clamp, Converter, Tonemapper};
use root::world::{EnvironmentMap, Material, MaterialEnum, TransportMode, NORMAL_OFFSET};
use root::*;

use log::LevelFilter;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use simplelog::{ColorChoice, CombinedLogger, TermLogger, TerminalMode, WriteLogger};
use structopt::StructOpt;

use sdfu::{Sphere, SDF};
use ultraviolet::vec::Vec3 as uvVec3;

#[inline(always)]
fn convert(p: Point3) -> uvVec3 {
    let [x, y, z, _] = p.as_array();
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
    pub config_file: String,
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

macro_rules! generate_primitive_enum {
    ($($variant:ident => $type:ty )+) => {
        #[derive(Copy, Clone)]
        enum PrimitiveEnum {
            $(
                $variant($type),
            )+
        }
        impl SDF<f32, uvVec3> for PrimitiveEnum {
            fn dist(&self, p: uvVec3) -> f32 {
                match self {
                    $(
                        PrimitiveEnum::$variant(inner) => {
                            inner.dist(p)
                        },
                    )+
                }
            }
        }
    };
}

generate_primitive_enum!(Sphere => Sphere<f32>);

struct Scene {
    // primitives: Vec<Box<dyn SDF<f32, uvVec3>>>,
    primitives: Vec<PrimitiveEnum>,
    environment: EnvironmentMap,
    materials: Vec<MaterialEnum>,
    // indexed by the same index as primitives, returns the material id for that primitive
    material_map: Vec<usize>,
    // max_depth: usize,
    world_aabb: AABB,
}

impl Scene {
    fn sdf(&self, p: Point3) -> (f32, usize) {
        let mut min_time = f32::INFINITY;
        let mut selected = 0;

        for (index, prim) in self.primitives.iter().enumerate() {
            let time = prim.dist(convert(p));

            if time < min_time {
                min_time = time;
                selected = index
            }
        }
        (min_time, selected)
    }
    fn normal(&self, p: Point3, index: usize, threshold: f32) -> Vec3 {
        // get normal from prim[index]
        Vec3(deconvert(
            self.primitives[index]
                .normals(threshold)
                .normal_at(convert(p)),
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
}

impl Scene {
    pub fn march(&self, r: Ray, threshold: f32, offset: f32, printout: bool) -> MarchResult {
        let mut distance = offset;
        let mut p = r.origin;
        let mut maybe_normal: Option<Vec3> = None;
        let mut closest_primitive: Option<usize> = None;
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
            let (current_distance, primitive) = self.sdf(p);
            if printout {
                println!(
                    "from point {:?}, time ({}) = {} + {} with nearest primitive = {}",
                    p,
                    distance + current_distance,
                    distance,
                    current_distance,
                    primitive
                );
            }
            distance += current_distance;
            if current_distance.abs() < threshold {
                // reached boundary,
                // update p
                p = r.point_at_parameter(distance);
                maybe_normal = Some(self.normal(p, primitive, 0.01));
                closest_primitive = Some(primitive);
                break;
            }
        }
        match maybe_normal {
            Some(normal) => MarchResult::SurfaceIntersection {
                point: p,
                normal,
                material: self.material_map
                    [closest_primitive.unwrap_or_else(|| panic!("strange error at point {:?}", p))],
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
        bounces: usize,
        sampler: &mut Box<dyn Sampler>,
        printout: bool,
    ) -> XYZColor {
        let mut throughput = SingleEnergy(1.0);
        let mut sum = SingleEnergy(0.0);
        for bounce in 0..bounces {
            match self.march(r, 0.001, 0.001, printout) {
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

                    // add if material.emission

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
                            if pdf.is_nan() || pdf.0 == 0.0 {
                                break;
                            }
                            throughput *= wo.z().abs() * f / pdf.0;
                            if throughput.0 == 0.0 {
                                break;
                            }
                            r = Ray::new(
                                point + normal * NORMAL_OFFSET * wo.z().signum(),
                                frame.to_world(&wo).normalized(),
                            );
                        }
                        None => {
                            warn_once!("didn't bounce at {:?}", point);
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

    let settings = get_settings(opt.config_file).unwrap();
    let (config, cameras) = parse_cameras_from(settings);
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
    let camera = cameras[0].with_aspect_ratio(height as f32 / width as f32);

    let world = construct_world(config.scene_file).unwrap();

    let mut primitives: Vec<PrimitiveEnum> = Vec::new();
    let mut material_map = Vec::new();

    // due to the culling optimization within construct world
    // assigning materials is tricky, currently.
    // you need to create dummy instances that reference materials so their data gets loaded
    // and you need to assign by finding that material through its type information here, i.e.
    primitives.push(PrimitiveEnum::Sphere(sdfu::Sphere::new(1.0)));
    // primitives.push(Box::new(sdfu::Sphere::new(1.0)));
    // primitives.push(Box::new(
    //     sdfu::Sphere::new(1.0).translate(uvVec3::new(0.0, 1.0, 1.0)),
    // ));

    // find the first lambertian
    world
        .materials
        .iter()
        .enumerate()
        .find(|(index, material)| matches!(material, MaterialEnum::Lambertian(_)))
        .and_then(|(i, material)| {
            material_map.push(i);
            Some(material)
        })
        .unwrap();
    // find the first diffuse light
    world
        .materials
        .iter()
        .enumerate()
        .find(|(index, material)| matches!(material, MaterialEnum::DiffuseLight(_)))
        .and_then(|(i, material)| {
            material_map.push(i);
            Some(material)
        })
        .unwrap();

    let env_map = world.environment;

    let mut world_aabb = AABB::new(Point3::new(-1.1, -1.1, -1.1), Point3::new(1.1, 1.1, 1.1));
    // world_aabb.grow_mut(&camera.origin);
    let (min, max) = {
        let aabb = camera.get_surface().unwrap().aabb();
        (aabb.min, aabb.max)
    };
    world_aabb.grow_mut(&min);
    world_aabb.grow_mut(&max);

    let scene = Scene {
        primitives,
        environment: env_map,
        materials: world.materials,
        material_map,
        // max_depth: 20,
        // world aabb needs to encompass camera
        world_aabb,
    };

    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;

    let mut render_film = Film::new(width, height, XYZColor::BLACK);

    // let converter = Converter::sRGB;

    // let mut tonemapper = Clamp::new(-1.0, false, true);
    let (mut tonemapper, converter) = parse_tonemapper(render_settings.tonemap_settings);

    let mut total_samples = 0;
    let samples_per_frame = 1;
    let bounces = 5;

    if opt.dry_run {
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let lambda = wavelength_bounds.sample(sampler.draw_1d().x);
        let (u, v) = (0.501, 0.5);
        let (r, pdf) = camera.get_ray(&mut sampler, lambda, u, v);

        let color = scene.color(r, lambda, bounces, &mut sampler, true);
        return;
    }

    root::window_loop(
        width,
        height,
        144,
        WindowOptions::default(),
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
    )
}
