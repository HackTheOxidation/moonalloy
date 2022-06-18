//! Methods - A collection of techniques in Numerical Linear Algebra

use crate::linalg::array::Array;
use crate::linalg::matrix::Matrix;
use std::ops::Range;

/// Returns the solution of a linear system of equations in the form: Ax = b.
/// It uses Gauss-Elimination to find a solution.
///
/// # Arguments
///
/// * `a` - a matrix containing all the coefficients in the system.
/// * `b` - a vector containing all the constants in the system.
///
/// # Examples
///
/// ```
/// // Create a n*n-dimensional coefficient matrix `a` and a vector of constants `b`
/// let a = Matrix::new(&mut [Array::from(&mut [3.0, 2.0]), Array::from(&mut [-6.0, 6.0])]);
/// let b = Array::from(&mut [7.0, 6.0]);
///
/// // Solve the system with Gauss-Elimination
/// assert_eq!(Array::from(&mut [-1.0, 2.0]), gauss_elimination(a, b));
/// ```
pub fn gauss_elimination(a: Matrix, b: Array) -> Array {
    let augmented = a.augment(b);
    let reduced_row = row_echelon_form(augmented);
    println!("reduced_row = {}", reduced_row);
    back_substitution(reduced_row)
}

/// Calculate a reduced row echelon form of an augmented matrix.
///
/// Based on pseudocode from
/// https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
fn row_echelon_form(augmented: Matrix) -> Matrix {
    let (m, n) = augmented.dimensions();
    let mut a = augmented;
    let mut h = 0;
    let mut k = 0;

    let abs = |num: f64| {
	if num < 0.0 {
	    return -1.0 * num;
	} else {
	    return num;
	}
    };

    while h < m && k < n {
	let i_max = argmax(h..m, a, k, &abs);

	if a[i_max][k] == 0.0 {
	    k += 1;
	} else {
	    a.swap_rows(h, i_max);
	    for i in (h + 1)..m {
		let f = a[i][k] / a[h][k];
		a[i][k] = 0.0;
		
		for j in (k + 1)..n {
		    a[i][j] = a[i][j] - a[h][j] * f;
		}
	    }
	    h += 1;
	    k += 1;
	}
    }
    
    a
}

/// Use back substitution on an augmented matrix in reduced row echelon form.
fn back_substitution(reduced: Matrix) -> Array {
    let (rows, cols) = reduced.dimensions();
    let mut x = Array::zeros(rows);
    let n = rows - 1;
    let k = cols - 1;
    let y = |index| reduced[index][k];

    x[n] = y(n) / reduced[n][n]; 
    
    for i in 0..rows {
	let mut kernel = 0.0;
	for j in (i + 1)..n {
	    kernel += reduced[i][j] * x[j];
	}
	x[i] = (y(i) - kernel) / reduced[i][i];
    }

    println!("x = {}", x);
    
    x
}

/// Select the argumentt hat yields the maximum output when applied to a function.
pub fn argmax(range: Range<usize>, a: Matrix, k: usize, f: &dyn Fn(f64) -> f64) -> usize {
    let mut max_arg = range.start;
    let mut max_out = f(a[max_arg][k]);

    for i in range {
	if max_out <= f(a[i][k]) {
	    max_arg = i;
	    max_out = f(a[i][k]);
	}
    }

    max_arg
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_row_echelon_form() {
	let a = Matrix::new(&mut [Array::from(&mut [3.0, 2.0]), Array::from(&mut [-6.0, 6.0])]);
        let b = Array::from(&mut [7.0, 6.0]);
	let augmented = a.augment(b);
	let expected = Matrix::new(&mut [Array::from(&mut [-6.0, 6.0, 6.0]), Array::from(&mut [0.0, 5.0, 10.0])]);

	let actual = row_echelon_form(augmented);
	assert_eq!(expected, actual);
    }

    #[test]
    fn test_backsubstitution() {
	let augmented = Matrix::new(&mut [Array::from(&mut [-6.0, 6.0, 6.0]), Array::from(&mut [0.0, 5.0, 10.0])]);

	let expected = Array::from(&mut [-1.0, 2.0]);

	let actual = back_substitution(augmented);
	assert_eq!(expected, actual);
    }

    #[test]
    fn test_gauss_elimination() {
        let expected = Array::from(&mut [-1.0, 2.0]);
        let a = Matrix::new(&mut [Array::from(&mut [3.0, 2.0]), Array::from(&mut [-6.0, 6.0])]);
        let b = Array::from(&mut [7.0, 6.0]);

        let actual = gauss_elimination(a, b);
        assert_eq!(expected, actual);
    }
}
