// use math::XYZColor;
extern crate rust_pathtracer as root;
use math::*;
use root::materials::{refract, Material, GGX};

use root::world::TransportMode;

pub fn balance(f: f32, g: f32) -> f32 {
    f / (f + g)
}

#[derive(Clone)]
pub enum Layer {
    Diffuse { color: SPD },
    Dielectric(GGX),
    HGMedium { g: f32, attenuation: SPD },
    None,
}

impl Layer {
    pub fn bsdf(
        &self,
        lambda: f32,
        wi: Vec3,
        wo: Vec3,
        transport_mode: TransportMode,
    ) -> (SingleEnergy, PDF) {
        match self {
            Layer::Diffuse { color } => {
                let cosine = wo.z();
                if cosine * wi.z() > 0.0 {
                    (
                        SingleEnergy::new(color.evaluate(lambda).min(1.0) / PI),
                        (cosine / PI).into(),
                    )
                } else {
                    (0.0.into(), 0.0.into())
                }
            }
            Layer::Dielectric(GGX) => GGX.bsdf(lambda, (0.0, 0.0), transport_mode, wi, wo),
            Layer::HGMedium { g, attenuation } => (0.0.into(), 0.0.into()),
            Layer::None => (0.0.into(), 0.0.into()),
        }
    }
    pub fn generate(
        &self,
        lambda: f32,
        wi: Vec3,
        sample: Sample2D,
        transport_mode: TransportMode,
    ) -> Option<Vec3> {
        match self {
            Layer::Diffuse { color } => Some(random_cosine_direction(sample)),
            Layer::Dielectric(GGX) => GGX.generate(lambda, (0.0, 0.0), transport_mode, sample, wi),
            Layer::HGMedium { g, attenuation } => None,
            Layer::None => None,
        }
    }
    /*fn transmit(&self, data: &LayerData, wo: Vector3) -> Option<Vector3> {
        let eta = Self::eta(data, wo)?;
        Vector3::new(0.0, 0.0, 1.0).refract(wo, 1.0 / eta.extract(0))
    }*/
    pub fn perfect_transmission(&self, lambda: f32, wo: Vec3) -> Option<Vec3> {
        match self {
            Layer::Dielectric(GGX) => {
                let eta = GGX.eta.evaluate(lambda);
                refract(wo, Vec3::Z, 1.0 / eta)
            }
            Layer::Diffuse { .. } => None,
            Layer::HGMedium { g, .. } => {
                if *g > -1.0 {
                    Some(-wo)
                } else {
                    None
                }
            }
            Layer::None => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CLMVertex {
    pub wi: Vec3,
    pub wo: Vec3,
    pub throughput: f32,
    pub path_pdf: f32,
    pub index: usize,
}
#[derive(Clone, Debug)]
pub struct CLMPath(pub Vec<CLMVertex>);

#[derive(Clone)]
pub struct CLM {
    // 0 is base layer, other layers are on top
    pub layers: Vec<Layer>,
    pub bounce_limit: usize,
}

impl CLM {
    pub fn new(layers: Vec<Layer>, bounce_limit: usize) -> Self {
        CLM {
            layers,
            bounce_limit,
        }
    }
    pub fn generate_short(
        &self,
        lambda: f32,
        mut wo: Vec3,
        transport_mode: TransportMode,
    ) -> CLMPath {
        let mut path = Vec::new();
        let num_layers = self.layers.len();
        let (mut index, direction) = if wo.z() > 0.0 {
            (num_layers - 1, -1)
        } else {
            (0, 1)
        };
        let mut throughput = 1.0;
        let mut path_pdf = 1.0;
        loop {
            let layer = &self.layers[index];
            let wi = match layer.perfect_transmission(lambda, wo) {
                Some(wi) => wi,
                None => {
                    break;
                }
            };

            path.push(CLMVertex {
                wi,
                wo,
                throughput,
                path_pdf,
                index,
            });

            if (index == 0 && direction == -1) || (index + 1 == num_layers && direction == 1) {
                break;
            }
            let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);
            throughput *= f.0;
            path_pdf *= pdf.0;

            index = (index as isize + direction) as usize;

            wo = -wi;
        }

        CLMPath(path)
    }
    pub fn generate(
        &self,
        lambda: f32,
        mut wi: Vec3,
        sampler: &mut dyn Sampler,
        transport_mode: TransportMode,
    ) -> CLMPath {
        let mut path = Vec::new();
        let num_layers = self.layers.len();
        let mut index = if wi.z() > 0.0 { num_layers - 1 } else { 0 };

        for _ in 0..self.bounce_limit {
            let layer = &self.layers[index];
            let wo = match layer.generate(lambda, wi, sampler.draw_2d(), transport_mode) {
                Some(wo) => wo,
                None => break,
            };

            println!("{:?} {:?}", wi, wo);

            path.push(CLMVertex {
                wi,
                wo,
                throughput: 0.0,
                path_pdf: 0.0,
                index,
            });

            let is_up = wo.z() > 0.0;

            if !is_up && index > 0 {
                index -= 1;
            } else if is_up && index + 1 < num_layers {
                index += 1;
            } else {
                break;
            }

            wi = -wo;
        }

        CLMPath(path)
    }
    pub fn bsdf(
        &self,
        lambda: f32,
        wi: Vec3,
        wo: Vec3,
        sampler: &mut dyn Sampler,
        transport_mode: TransportMode,
    ) -> (SingleEnergy, PDF) {
        let opposite_mode = match transport_mode {
            TransportMode::Importance => TransportMode::Radiance,
            TransportMode::Radiance => TransportMode::Importance,
        };
        let long_path = self.generate(lambda, wi, sampler, transport_mode);
        println!("long path finished");
        // let wo = long_path.0.last().unwrap().wo;
        let short_path = self.generate_short(lambda, wo, opposite_mode);
        // let num_layers = self.layers.len();

        let mut sum = 0.0;
        let mut pdf_sum = 0.0;

        let nee_direction = if wo.z() > 0.0 { 1 } else { -1 };

        let mut throughput = 1.0;
        let mut path_pdf = 1.0;

        for vert in long_path.0.iter() {
            let index = vert.index;
            let layer = &self.layers[index];
            let nee_index = index as isize + nee_direction;

            if nee_index < 0 || nee_index as usize >= self.layers.len() {
                let nee_wo = if nee_index < 0 {
                    short_path.0.first().unwrap().wo
                } else {
                    short_path.0.last().unwrap().wo
                };
                let (left_f, left_path_pdf) = (throughput, path_pdf);

                let (left_connection_f, left_connection_pdf) =
                    layer.bsdf(lambda, vert.wi, nee_wo, transport_mode);

                let (total_throughput, total_path_pdf) = (
                    left_f * left_connection_f.0,
                    left_path_pdf * left_connection_pdf.0,
                );

                let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);

                let weight = balance(left_connection_pdf.0, pdf.0);

                if total_path_pdf > 0.0 {
                    sum += weight * total_throughput / total_path_pdf;
                    pdf_sum += total_path_pdf;
                }

                throughput *= (1.0 - weight) * f.0;
                path_pdf *= pdf.0
            } else {
                let nee_index = nee_index as usize;
                let nee_layer = &self.layers[nee_index];
                let nee_vert = short_path.0[short_path.0.len() - nee_index];

                let (left_f, left_path_pdf) = (throughput, path_pdf);

                let (right_f, right_path_pdf) = (nee_vert.throughput, nee_vert.path_pdf);

                let (left_connection_f, left_connection_pdf) =
                    layer.bsdf(lambda, vert.wi, -nee_vert.wi, transport_mode);
                let (right_connection_f, right_connection_pdf) =
                    nee_layer.bsdf(lambda, nee_vert.wi, nee_vert.wo, transport_mode);

                let (total_throughput, total_path_pdf) = (
                    left_f * left_connection_f.0 * right_connection_f.0 * right_f,
                    left_path_pdf * left_connection_pdf.0 * right_connection_pdf.0 * right_path_pdf,
                );

                let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);

                let weight = balance(
                    left_connection_pdf.0 * right_connection_pdf.0 * right_path_pdf,
                    pdf.0,
                );

                if total_path_pdf > 0.0 {
                    sum += weight * total_throughput / total_path_pdf;
                    pdf_sum += total_path_pdf;
                }

                throughput *= (1.0 - weight) * f.0;
                path_pdf *= pdf.0
            }
        }
        (sum.into(), pdf_sum.into())
    }
}

// impl Material for CLM {}

fn main() {
    use root::curves;
    use root::parsing::curves::load_multiple_csv_rows;

    let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));

    let glass = curves::cauchy(1.5, 10000.0);
    let flat_zero = curves::void();
    let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0);

    let cornell_colors = load_multiple_csv_rows(
        "data/curves/csv/cornell.csv",
        3,
        InterpolationMode::Cubic,
        |x| x,
        |y| y,
    )
    .expect("data/curves/csv/cornell.csv was not formatted correctly");
    let mut iter = cornell_colors.iter();
    let (cornell_white, cornell_green, cornell_red) = (
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
    );

    let clm = CLM::new(
        vec![
            Layer::Diffuse {
                color: cornell_white,
            },
            Layer::Dielectric(ggx_glass),
        ],
        20,
    );

    let path = clm.generate(
        500.0,
        Vec3::new(1.0, 1.0, 10.0).normalized(),
        &mut *sampler,
        TransportMode::Importance,
    );
    println!("long path finished");

    println!("{:?}", path);

    let wo = path.0.last().unwrap().wo;

    let (f, pdf) = clm.bsdf(
        500.0,
        Vec3::new(1.0, 1.0, 10.0).normalized(),
        wo,
        &mut *sampler,
        TransportMode::Importance,
    );
    println!("{}, {}", f.0, pdf.0);
}
