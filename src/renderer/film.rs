#[derive(Clone)]
pub struct Film
{
    pub ty: FilmType,
    pub window_function: WindowFunction,
    pub data: Vec2D<XYZColor>,
}


