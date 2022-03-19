use crate::linalg::array::Array;
use crate::machine_learning::model::Model;

pub fn evaluate_simple_linear_regression(observations: Array, xs: Array, model: &mut dyn Model) -> f64 {
    let predictions = model.predict(xs);
    
    if observations.len() == predictions.len() {
	let cos = cos_angle(observations, predictions);
	if cos < 0.0 {
	    cos * -1.0
	} else {
	    cos
	}
    } else {
	panic!("Error: Arrays lengths differ.");
    } 
}

fn cos_angle(v1: Array, v2: Array) -> f64 {
    v1.dotp(&v2) / (v1.norm() * v2.norm())
}
