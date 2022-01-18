use crate::linalg::array::Array;
use crate::linalg::matrix::Matrix;

pub fn gauss_elimination(a: Matrix, b: Array) -> Array {
    let (rows, _) = a.dimensions();
    for row in 0..(rows - 1) {
        for i in (row + 1)..rows {
            if a.get(i, row) != 0.0 {
                let factor = a.get(i, row) / a.get(row, row);
                let val = a
                    .splice(i, row + 1, rows)
                    .sub(&a.splice(row, row + 1, rows).scalar(factor));
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
