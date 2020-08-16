#![feature(clamp, slice_fill, vec_remove_item)]

// #[macro_use]
extern crate packed_simd;

pub mod lens_element;

use lens_element::*;

pub struct LensAssembly {
    pub lenses: Vec<LensInterface>,
    pub aperture_index: usize,
}

impl LensAssembly {
    pub fn new(self, lenses: &[LensInterface]) -> Self {
        // returns last index if slice does not contain an aperture
        let mut i = 0;
        for elem in lenses {
            if elem.lens_type == LensType::Aperture {
                break;
            }
            i += 1;
        }
        LensAssembly {
            lenses: lenses.into(),
            aperture_index: i,
        }
    }
    pub fn aperture_radius(self) -> f32 {
        let aperture_index = self.aperture_index;
        self.lenses[aperture_index].housing_radius
    }
    pub fn aperture_position(self, zoom: f32) -> f32 {
        // returns the end if there is no aperture
        let mut pos = 0.0;
        for elem in self.lenses.iter() {
            if elem.lens_type == LensType::Aperture {
                break;
            }
            pos += elem.thickness_at(zoom);
        }
        pos
    }
    pub fn total_thickness_at(self, zoom: f32) -> f32 {
        let mut pos = 0.0;
        for elem in self.lenses.iter() {
            pos += elem.thickness_at(zoom);
        }
        pos
    }
}
