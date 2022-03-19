use crate::linalg::array::Array;
use crate::machine_learning::model::Model;

///
pub struct SimpleLinearRegression {
    slope: f64,
    feature: f64,
}

impl SimpleLinearRegression {
    pub fn new() -> Self {
	SimpleLinearRegression {
	    slope: 0.0,
	    feature: 0.0,
	}
    }

    pub fn from(slope: f64, feature: f64) -> Self {
	SimpleLinearRegression {
	    slope,
	    feature,
	}
    }

    pub fn get_slope(&self) -> f64 {
	self.slope
    }

    pub fn get_feature(&self) -> f64 {
	self.feature
    }
}

impl Model for SimpleLinearRegression {
    fn predict(&mut self, observed_xs: Array) -> Array {
	observed_xs.scalar_mult(self.slope).scalar_add(self.feature)
    }

    fn optimize(&mut self, xs: Array, ys: Array) {
	self.slope = xs.scalar_sub(xs.average()).mult(&ys.scalar_sub(ys.average())).sum() /
	    xs.scalar_sub(xs.average()).mult(&xs.scalar_sub(xs.average())).sum();
	self.feature = ys.average() - (self.slope * xs.average());
    }
}
