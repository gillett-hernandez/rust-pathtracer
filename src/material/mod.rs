mod bxdf;
mod pdf;

pub use bxdf::BRDF;
pub use pdf::PDF;

use std::marker::{Send, Sync};

pub trait Material: PDF + BRDF + Send + Sync {}
