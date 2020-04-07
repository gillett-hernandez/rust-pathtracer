mod bxdf;
mod pdf;

pub use bxdf::BRDF;
pub use pdf::PDF;

pub trait Material: PDF + BRDF {}
