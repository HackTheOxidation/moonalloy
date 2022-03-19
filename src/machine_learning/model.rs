use crate::linalg::array::Array;

pub trait Model {
    fn optimize(&mut self, xs: Array, ys: Array);
    fn predict(&mut self, observed_xs: Array) -> Array;
}
