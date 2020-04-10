pub struct Transform3 {
    matrix: [f32; 16],
}
pub trait Transformable {
    fn transform_in_place(&mut self, transform: Transform3);
    fn transform(mut self, transform: Transform3) -> Self;
}
