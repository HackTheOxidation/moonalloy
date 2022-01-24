//! Methods - A collection of techniques in Numerical Linear Algebra

use crate::linalg::array::Array;
use crate::linalg::matrix::Matrix;

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
/// assert_eq!(Array::from(&mut [1.0, 2.0]), gauss_elimination(a, b));
/// ```
pub fn gauss_elimination(mut a: Matrix, mut b: Array) -> Array {
    let (rows, _) = a.dimensions();
    for row in 0..(rows - 1) {
        for i in (row + 1)..rows {
            if a.get(i, row) != 0.0 {
                let factor = a.get(i, row) / a.get(row, row);
                let val = a
                    .splice(i, row + 1, rows)
                    .minus(&a.splice(row, row + 1, rows).scalar(factor));
                a.set_row(val, i);
                b.set(b.get(i) - factor * b.get(row), i);
            }
        }
    }

    for k in (0..rows).rev() {
        let val = (b.get(k) - a.splice(k, k + 1, rows).dotp(&b.splice(k + 1, rows))) / a.get(k, k);
        b.set(val, k);
    }

    return b;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gauss_simple() {
        let expected = Array::from(&mut [1.0, 2.0]);
        let a = Matrix::new(&mut [Array::from(&mut [3.0, 2.0]), Array::from(&mut [-6.0, 6.0])]);
        let b = Array::from(&mut [7.0, 6.0]);

        let actual = gauss_elimination(a, b);
        assert_eq!(expected, actual);
    }

    #[test]
    fn gauss_multiple() {
        let expected = Array::from(&mut [2.0, 3.0, -1.0]);
        let a = Matrix::new(&mut [
            Array::from(&mut [2.0, 1.0, -1.0]),
            Array::from(&mut [-3.0, -1.0, 2.0]),
            Array::from(&mut [-2.0, 1.0, 2.0]),
        ]);
        let b = Array::from(&mut [8.0, -11.0, -3.0]);

        let actual = gauss_elimination(a, b);
        assert_eq!(expected, actual);
    }
}
