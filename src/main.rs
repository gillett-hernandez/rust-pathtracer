// #![allow(unused_imports, unused_variables, unused)]
#![feature(clamp, slice_fill, vec_remove_item)]

extern crate image;

#[macro_use]
extern crate packed_simd;

pub mod aabb;
pub mod accelerator;
pub mod camera;
pub mod config;
pub mod curves;
pub mod geometry;
pub mod hittable;
pub mod integrator;
pub mod material;
pub mod materials;
pub mod math;
pub mod parsing;
pub mod renderer;
pub mod tonemap;
pub mod world;

use camera::*;
use config::{get_settings, Config};
use geometry::*;
use math::*;
use parsing::*;
use renderer::{NaiveRenderer, Renderer};
use world::*;

pub const NORMAL_OFFSET: f32 = 0.00001;
pub const INTERSECTION_TIME_OFFSET: f32 = 0.000001;

fn parse_cameras_from(settings: &Config) -> Vec<Camera> {
    let mut cameras = Vec::<Camera>::new();
    for camera_config in &settings.cameras {
        let camera: Camera = match camera_config {
            config::CameraSettings::SimpleCamera(cam) => {
                let shutter_open_time = cam.shutter_open_time.unwrap_or(0.0);
                Camera::ProjectiveCamera(ProjectiveCamera::new(
                    Point3::from(cam.look_from),
                    Point3::from(cam.look_at),
                    Vec3::from(cam.v_up.unwrap_or([0.0, 0.0, 1.0])),
                    cam.vfov,
                    1.0,
                    cam.focal_distance.unwrap_or(10.0),
                    cam.aperture_size.unwrap_or(0.0),
                    shutter_open_time,
                    cam.shutter_close_time.unwrap_or(1.0).max(shutter_open_time),
                ))
            }
        };
        cameras.push(camera);
    }
    cameras
}

#[allow(dead_code)]
fn white_furnace_test(material: MaterialEnum) -> World {
    let world = World::new(
        vec![Instance::new(
            Aggregate::from(Sphere::new(5.0, Point3::ORIGIN)),
            None,
            MaterialId::Material(0),
            0,
        )],
        vec![material],
        EnvironmentMap::Constant {
            color: curves::cie_e(1.0),
            strength: 1.0,
        },
        1.0,
    );
    world
}

#[allow(unused_variables)]
fn cornell_box(
    additional_instances: Vec<Instance>,
    additional_materials: Vec<MaterialEnum>,
    env_map: EnvironmentMap,
    env_sampling_probability: f32,
) -> World {
    // let red = curves::red(1.0);
    // let green = curves::green(1.0);
    let blue = curves::blue(1.0);
    // let white = curves::cie_e(1.0);
    let cornell_colors = load_csv("data/curves/csv/cornell.csv", 3, |x| x)
        .expect("data/curves/csv/cornell.csv was not formatted correctly");
    let mut iter = cornell_colors.iter();
    let (cornell_white, cornell_green, cornell_red) = (
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
    );
    let lambertian_white = MaterialEnum::from(Lambertian::new(cornell_white));
    let lambertian_red = MaterialEnum::from(Lambertian::new(cornell_red));
    let lambertian_green = MaterialEnum::from(Lambertian::new(cornell_green));
    let _lambertian_blue = MaterialEnum::from(Lambertian::new(blue));

    let mut world_materials = vec![lambertian_white, lambertian_green, lambertian_red];

    // let mut world_instances = vec![

    //     Instance::from(Aggregate::from(AARect::new(
    //         // left wall
    //         (2.0, 2.0),
    //         Point3::new(0.0, 1.0, 0.0),
    //         Axis::Y,
    //         true,
    //         1.into(),
    //         2,
    //     ))),
    //     Instance::from(Aggregate::from(AARect::new(
    //         // right wall
    //         (2.0, 2.0),
    //         Point3::new(0.0, -1.0, 0.0),
    //         Axis::Y,
    //         true,
    //         2.into(),
    //         3,
    //     ))),
    //     Instance::from(Aggregate::from(AARect::new(
    //         // back wall
    //         (2.0, 2.0),
    //         Point3::new(1.0, 0.0, 0.0),
    //         Axis::X,
    //         true,
    //         0.into(),
    //         4,
    //     ))),
    // ];
    let mut world_instances = vec![
        Instance::new(
            //ceiling
            Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 0.0, 1.0),
                Axis::Z,
                true,
            )),
            None,
            0.into(),
            0,
        ),
        Instance::new(
            // floor
            Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 0.0, -1.0),
                Axis::Z,
                true,
            )),
            None,
            0.into(),
            1,
        ),
        Instance::new(
            Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, 1.0, 0.0),
                Axis::Y,
                true,
            )),
            None,
            1.into(),
            2,
        ),
        Instance::new(
            Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(0.0, -1.0, 0.0),
                Axis::Y,
                true,
            )),
            None,
            2.into(),
            3,
        ),
        Instance::new(
            Aggregate::from(AARect::new(
                (2.0, 2.0),
                Point3::new(1.0, 0.0, 0.0),
                Axis::X,
                true,
            )),
            None,
            0.into(),
            4,
        ),
    ];

    let base_length = world_materials.len();
    world_materials.extend_from_slice(additional_materials.as_slice());
    println!(
        "extended world materials list by {}",
        world_materials.len() - base_length
    );
    for mut instance in additional_instances {
        instance.material_id = match instance.material_id {
            MaterialId::Material(old) => MaterialId::Material((base_length + old as usize) as u16),
            MaterialId::Light(old) => MaterialId::Light((base_length + old as usize) as u16),
            MaterialId::Camera(old) => {
                continue;
            }
        };
        instance.instance_id = world_instances.len();
        println!("adding instance to world: {:?}", instance);
        world_instances.push(instance);
    }

    let world = World::new(
        world_instances,
        world_materials,
        env_map,
        env_sampling_probability,
    );
    world
}

#[allow(unused_variables)]
fn construct_scene(config: &Config) -> World {
    // load some curves
    let (silver_ior, silver_kappa) =
        load_ior_and_kappa("data/curves/csv/silver.csv", |x: f32| x * 1000.0).unwrap();
    let (gold_ior, gold_kappa) =
        load_ior_and_kappa("data/curves/csv/gold.csv", |x: f32| x * 1000.0).unwrap();

    let (bismuth_ior, bismuth_kappa) =
        load_ior_and_kappa("data/curves/csv/bismuth.csv", |x: f32| x * 1000.0).unwrap();
    let (copper_ior, copper_kappa) =
        load_ior_and_kappa("data/curves/csv/copper-mcpeak.csv", |x: f32| x * 1000.0).unwrap();
    let (lead_ior, lead_kappa) =
        load_ior_and_kappa("data/curves/csv/lead.csv", |x: f32| x * 1000.0).unwrap();
    let (cold_lead_ior, cold_lead_kappa) =
        load_ior_and_kappa("data/curves/csv/lead-140K.csv", |x: f32| x * 1000.0).unwrap();
    let (platinum_ior, platinum_kappa) =
        load_ior_and_kappa("data/curves/csv/platinum.csv", |x: f32| x * 1000.0).unwrap();
    let (iron_ior, iron_kappa) =
        load_ior_and_kappa("data/curves/csv/iron-johnson.csv", |x: f32| x * 1000.0).unwrap();

    // create some curves
    let white = curves::cie_e(1.0);
    let flat_zero = curves::void();
    let flat_one = curves::cie_e(1.0);
    let cie_e_illuminant = curves::cie_e(15.0);
    let red = curves::red(1.0);
    let green = curves::green(1.0);
    let blue = curves::blue(1.0);
    let white = curves::cie_e(1.0);
    let moissanite = curves::cauchy(2.5415, 28665.0);
    let glass = curves::cauchy(1.45, 3540.0);

    // create materials
    let lambertian_white = MaterialEnum::from(Lambertian::new(white));
    let lambertian_red = MaterialEnum::from(Lambertian::new(red));
    let lambertian_green = MaterialEnum::from(Lambertian::new(green));
    let lambertian_blue = MaterialEnum::from(Lambertian::new(blue));

    let ggx_glass =
        MaterialEnum::from(GGX::new(0.0001, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_glass_rough =
        MaterialEnum::from(GGX::new(0.2, glass.clone(), 1.0, flat_zero.clone(), 1.0));
    let ggx_moissanite =
        MaterialEnum::from(GGX::new(0.0001, moissanite, 1.0, flat_zero.clone(), 1.0));
    let ggx_unrealistic_dispersion = MaterialEnum::from(GGX::new(
        0.001,
        curves::cauchy(1.45, 50000.0),
        1.0,
        flat_zero.clone(),
        1.0,
    ));
    let ggx_silver_metal = MaterialEnum::from(GGX::new(
        0.003,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_copper_metal = MaterialEnum::from(GGX::new(
        0.003,
        copper_ior.clone(),
        1.0,
        copper_kappa.clone(),
        0.0,
    ));
    let ggx_silver_metal_rough = MaterialEnum::from(GGX::new(
        0.08,
        silver_ior.clone(),
        1.0,
        silver_kappa.clone(),
        0.0,
    ));
    let ggx_gold_metal = MaterialEnum::from(GGX::new(0.0003, gold_ior, 1.0, gold_kappa, 0.0));
    let ggx_lead_metal = MaterialEnum::from(GGX::new(0.0003, lead_ior, 1.0, lead_kappa, 0.0));
    let ggx_cold_lead_metal =
        MaterialEnum::from(GGX::new(0.0003, cold_lead_ior, 1.0, cold_lead_kappa, 0.0));
    let ggx_platinum_metal =
        MaterialEnum::from(GGX::new(0.0003, platinum_ior, 1.0, platinum_kappa, 0.0));
    let ggx_bismuth_metal =
        MaterialEnum::from(GGX::new(0.0008, bismuth_ior, 1.0, bismuth_kappa, 0.0));
    let ggx_iron_metal = MaterialEnum::from(GGX::new(0.0003, iron_ior, 1.0, iron_kappa, 0.0));

    let additional_instances = vec![
        Instance::new(
            Aggregate::from(Disk::new(0.7, Point3::new(0.0, 0.0, 0.0), false)),
            Some(Transform3::from_stack(
                Some(Transform3::from_scale(Vec3::new(-1.0, -1.0, -1.0))),
                None,
                Some(Transform3::from_translation(
                    Point3::ORIGIN - Point3::new(0.0, 0.0, 0.95),
                )),
            )),
            MaterialId::Light(0),
            0,
        ),
        Instance::new(
            Aggregate::from(Sphere::new(
                // ball in front
                0.3,
                Point3::new(-0.1, 0.0, -0.6),
            )),
            None,
            1.into(),
            1,
        ),
        Instance::new(
            Aggregate::from(Sphere::new(
                // ball on back right
                0.3,
                Point3::new(0.5, -0.35, -0.7),
            )),
            None,
            2.into(),
            3,
        ), // ball at origin
        Instance::new(
            Aggregate::from(Sphere::new(
                // ball on back left
                0.3,
                Point3::new(0.5, 0.35, -0.7),
            )),
            None,
            3.into(),
            4,
        ),
    ]; // ball at origin

    // create some illuminants and lights
    let blackbody_illuminant1_dim = curves::blackbody(2700.0, 1.0);
    let blackbody_illuminant1 = curves::blackbody(2700.0, 100.0);
    let blackbody_illuminant1_bright = curves::blackbody(2700.0, 500.0);
    let blackbody_illuminant2 = curves::blackbody(4500.0, 18.0);
    let xenon_lamp = curves::spectra("data/curves/spectra/xenon_lamp.spectra", 20.0);
    let cie_e_illuminant_low_power = curves::cie_e(0.25);

    let light_material =
        MaterialEnum::from(DiffuseLight::new(blackbody_illuminant2, Sidedness::Forward));
    // parallel light with low sharpness should approximate a diffuse light
    // let light_material = MaterialEnum::from(ParallelLight::new(
    //     blackbody_illuminant2,
    //     0.0,
    //     Sidedness::Forward,
    // ));
    // let light_material = MaterialEnum::from(SharpLight::new(
    //     blackbody_illuminant2,
    //     9.0,
    //     Sidedness::Forward,
    // ));

    let additional_materials = vec![
        light_material,
        // lambertian_red,
        // ggx_moissanite,
        ggx_unrealistic_dispersion,
        // ggx_glass,
        ggx_gold_metal,
        // lambertian_green,
        ggx_iron_metal,
        // lambertian_red,
    ];
    // let additional_materials = vec![
    //     light_material,
    //     lambertian_blue,
    //     lambertian_green,
    //     lambertian_red,
    // ];
    let world_illuminant = blackbody_illuminant1_dim;

    let env_map = EnvironmentMap::Constant {
        color: world_illuminant,
        strength: config.env_strength.unwrap_or(1.0),
    };

    // let env_map = EnvironmentMap::Sun {
    //     color: world_illuminant,
    //     strength: config.env_strength.unwrap_or(1.0),
    //     solid_angle: 0.14,
    //     sun_direction: Vec3::Z,
    // };

    cornell_box(
        additional_instances,
        additional_materials,
        env_map,
        config.env_sampling_probability.unwrap_or(0.5),
    )
}

fn main() -> () {
    let config: Config = match get_settings("data/config.toml".to_string()) {
        Ok(expr) => expr,
        Err(v) => {
            println!("{:?}", "couldn't read config.toml");
            println!("{:?}", v);
            return;
        }
    };
    let threads = config
        .render_settings
        .iter()
        .map(|i| &i.threads)
        .fold(1, |a, &b| a.max(b.unwrap_or(1)));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();

    let world = construct_scene(&config);

    let cameras: Vec<Camera> = parse_cameras_from(&config);
    // some integrators only work with certain renderers.
    // collect the render settings bundles that apply to certain integrators, and correlate them with their corresponding renderers.
    // use multiple renderers if necessary

    // let renderer = construct_renderer(&config);
    let renderer = NaiveRenderer::new();
    renderer.render(world, cameras, &config);
}
