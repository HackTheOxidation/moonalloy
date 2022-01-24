//! Matrix - An implementation of a mathematical Matrix
//!
//! This module contains structures and functions for manipulating matrices in Linear Algebra.
//! All of the basics of matrix arithmetic.

use crate::Array;

use std::alloc::{alloc, Layout};
use std::fmt::*;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

/// A representation of a mathematical matrix
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Matrix {
    /// Number of rows in the matrix
    rows: usize,
    /// Number of columns in the matrix
    cols: usize,
    /// Elements of the matrix as a raw pointer of Arrays
    arrays: *mut Array,
}

impl Matrix {
    /// Checks that the slice of Arrays can be converted to a valid matrix.
    ///
    /// # Arguments
    ///
    /// * `slice` - a mutable slice of Arrays
    fn is_valid_slice(slice: &mut [Array]) -> bool {
        let len = slice[0].len();
        for i in 1..slice.len() {
            assert!(len == slice[i].len());
        }

        true
    }

    /// Returns a new matrix from a mutable slice of Arrays.
    ///
    /// # Arguments
    ///
    /// * `slice` - a mutable slice of Arrays
    ///
    /// # Panics
    ///
    /// The slice must adhere to the `n`x`m` form of a matrix otherwise
    /// the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix
    /// use moonalloy::linalg::matrix::Matrix;
    /// let mat = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0)]);
    /// ```
    pub fn new(slice: &mut [Array]) -> Matrix {
        assert!(Matrix::is_valid_slice(slice));
        Matrix {
            rows: slice.len(),
            cols: slice[0].len(),
            arrays: slice.as_mut_ptr(),
        }
    }

    /// Returns a new matrix where all the elements have the same value
    ///
    /// # Arguments
    ///
    /// * `val` - the value for all the elements in the matrix.
    /// * `rows` - the number of rows in the new matrix.
    /// * `cols` - the number of columns in the new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix where all elements have the value 3.0
    /// use moonalloy::linalg::matrix::Matrix;
    /// let mat = Matrix::of(3.0, 2, 2);
    ///
    /// assert_eq(Matrix::new(&mut [Array::from(&mut [3.0, 3.0]), Array::from(&mut [3.0, 3.0)]), mat);
    /// ```
    fn of(val: f64, rows: usize, cols: usize) -> Matrix {
        let mat_slice = unsafe {
            let layout = Layout::array::<Array>(rows).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, rows)
        };

        for i in 0..rows {
            mat_slice[i] = Array::of(val, cols);
        }

        Matrix {
            rows,
            cols,
            arrays: mat_slice.as_mut_ptr(),
        }
    }

    /// Returns a new matrix where all the elements have the value of 0.0
    ///
    /// # Arguments
    ///
    /// * `rows` - the number of rows in the new matrix.
    /// * `cols` - the number of columns in the new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix where all elements have the value 0.0
    /// use moonalloy::linalg::matrix::Matrix;
    /// let mat = Matrix::zeros(2, 2);
    ///
    /// assert_eq(Matrix::new(&mut [Array::from(&mut [0.0, 0.0]), Array::from(&mut [0.0, 0.0)]), mat);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::of(0.0, rows, cols)
    }

    /// Returns a new matrix where all the elements have the value of 1.0
    ///
    /// # Arguments
    ///
    /// * `rows` - the number of rows in the new matrix.
    /// * `cols` - the number of columns in the new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix where all elements have the value 1.0
    /// use moonalloy::linalg::matrix::Matrix;
    /// let mat = Matrix::ones(2, 2);
    ///
    /// assert_eq(Matrix::new(&mut [Array::from(&mut [1.0, 1.0]), Array::from(&mut [1.0, 1.0)]), mat);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix::of(1.0, rows, cols)
    }

    /// Returns an `n`x`n` identity matrix.
    ///
    /// # Arguments
    ///
    /// * `len` - the rank of the identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 identity matrix
    /// use moonalloy::linalg::matrix::Matrix;
    /// let mat = Matrix::identity(2);
    ///
    /// assert_eq(Matrix::new(&mut [Array::from(&mut [1.0, 0.0]), Array::from(&mut [0.0, 1.0)]), mat);
    /// ```
    pub fn identity(len: usize) -> Matrix {
        let mat = Matrix::zeros(len, len);

        let mat_slice = unsafe { std::slice::from_raw_parts_mut(mat.arrays, len) };

        for i in 0..len {
            let slice = &mut mat_slice[i];

            slice[i] = 1.0;
        }

        mat
    }

    /// Returns a string representation of a matrix.
    pub fn to_string(&self) -> String {
        let array_slice =
            unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        let mut result = String::from("Matrix: \n[");

        for (i, arr) in array_slice.iter().enumerate() {
            let slice = arr.as_slice();

            result.push_str(format!("{:?}", slice).as_str());
            if i < arr.len() - 1 {
                result.push_str(", \n ")
            }
        }

        result.push(']');

        result
    }

    /// Adds two matrices without modifying the originals.
    ///
    /// # Arguments
    ///
    /// * `other` - the other matrix to add.
    ///
    /// # Panics
    ///
    /// The two matrices must have the same dimensions in order to add them
    /// together. If not, the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create to matrices `a` and `b` and add them.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    /// let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [3.0, 5.0]), Array::from(&mut [8.0, 13.0])]), a.plus(&b));
    /// // Use the `+`-operator as a shorthand for this.
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [3.0, 5.0]), Array::from(&mut [8.0, 13.0])]), a + b);
    /// ```
    pub fn plus(&self, other: &Matrix) -> Matrix {
        assert!(
            self.cols == other.cols,
            "ERROR - Matrix addition: Columns differ in dimensions."
        );
        assert!(
            self.rows == other.rows,
            "ERROR - Matrix addition: Rows differ in dimensions."
        );

        let result = unsafe {
            let layout = Layout::array::<Array>(self.rows).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.rows)
        };

        let mat_slice1 = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        let mat_slice2 =
            unsafe { std::slice::from_raw_parts_mut(other.arrays, other.rows) };

        for i in 0..self.rows {
            result[i] = mat_slice1[i].plus(&mat_slice2[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Multiply every element in a matrix with a scalar value without modifying the original.
    ///
    /// # Arguments
    ///
    /// * `scal` - the scalar value to multiply with.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a matrix `a` and multiply all elements with -1.0.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [-1.0, -2.0]), Array::from(&mut [-3.0, -5.0])]), a.scalar(-1.0));
    /// // Use the unary `-`-operator as a shorthand for multiplication with -1.0.
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [-1.0, -2.0]), Array::from(&mut [-3.0, -5.0])]), -a);
    /// ```
    pub fn scalar(&self, scal: f64) -> Matrix {
        let result = unsafe {
            let layout = Layout::array::<Array>(self.rows).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.rows)
        };

        let slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        for i in 0..self.rows {
            result[i] = slice[i].scalar(scal);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Subtracts two matrices without modifying the originals.
    ///
    /// # Arguments
    ///
    /// * `other` - the other matrix to subtract.
    ///
    /// # Panics
    ///
    /// The two matrices must have the same dimensions in order to subtract them.
    /// If not, the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create two matrices `a` and `b` and subtract them.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    /// let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [-1.0, -1.0]), Array::from(&mut [-2.0, -3.0])]), a.minus(&b));
    /// // Use the `+`-operator as a shorthand for this.
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [-1.0, -1.0]), Array::from(&mut [-2.0, -3.0])]), a - b);
    /// ```
    pub fn minus(&self, other: &Matrix) -> Matrix {
        assert!(
            self.cols == other.cols,
            "ERROR - Matrix subtraction: Columns differ in dimensions."
        );
        assert!(
            self.rows == other.rows,
            "ERROR - Matrix subtraction: Rows differ in dimensions."
        );

        let result = unsafe {
            let layout = Layout::array::<Array>(self.rows).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.rows)
        };

        let mat_slice1 = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        let mat_slice2 =
            unsafe { std::slice::from_raw_parts_mut(other.arrays, other.rows) };

        for i in 0..self.rows {
            result[i] = mat_slice1[i].minus(&mat_slice2[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Multiplies two matrices element by element without modifying the originals.
    ///
    /// # Arguments
    ///
    /// * `other` - the other matrix to perform element-wise multiplication with.
    ///
    /// # Panics
    ///
    /// The two matrices must have the same dimensions in order to multiply them.
    /// If not, the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create two matrices `a` and `b` and perform element-wise multiplication.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    /// let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [2.0, 6.0]), Array::from(&mut [15.0, 40.0])]), a.elem_mult(&b));
    /// // Use the `*`-operator as a shorthand for this.
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [2.0, 6.0]), Array::from(&mut [15.0, 40.0])]), a * b);
    /// ```
    pub fn elem_mult(&self, other: &Matrix) -> Matrix {
        assert!(
            self.cols == other.cols,
            "ERROR - Matrix element-wise multiplication: Columns differ in dimensions."
        );
        assert!(
            self.rows == other.rows,
            "ERROR - Matrix element-wise multiplication: Rows differ in dimensions."
        );

        let result = unsafe {
            let layout = Layout::array::<Array>(self.rows).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.rows)
        };

        let mat_slice1 = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        let mat_slice2 =
            unsafe { std::slice::from_raw_parts_mut(other.arrays, other.rows) };

        for i in 0..self.rows {
            result[i] = mat_slice1[i].mult(&mat_slice2[i]);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Returns a transpose of a matrix without modifying the original.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a matrix `a` and transpose it.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [1.0, 3.0]), Array::from(&mut [2.0, 5.0])]), a.transpose());
    /// ```
    pub fn transpose(&self) -> Matrix {
        let result = unsafe {
            let layout = Layout::array::<Array>(self.cols).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.cols)
        };

        let arr_slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        for i in 0..self.cols {
            result[i] = Array::zeros(self.rows);

            for j in 0..self.rows {
                result[i].set(arr_slice[j].get(i), j);
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Perform matrix multiplication on two matrices
    ///
    /// # Arguments
    ///
    /// * `other` - the other matrix to multiply with.
    ///
    /// # Panics
    ///
    /// For matrix multiplication of two matrices, A and B,
    /// A must have the dimensions `n`x`m` and B must have the dimensions `r`x`n`
    /// in order for the multiplication to be valid. 
    ///
    /// # Examples
    ///
    /// ```
    /// // Create two matrices `a` and `b` and multiply them.
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    /// let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
    ///
    /// assert_eq!(Matrix::new(&mut [Array::from(&mut [10.0, 19.0]), Array::from(&mut [21.0, 50.0])]), a.mult(&b));
    /// ```
    pub fn mult(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.cols,
            "ERROR - Matrix multiplication: Invalid dimensions."
        );

        let result = unsafe {
            let layout = Layout::array::<Array>(self.cols).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut Array, self.cols)
        };

        let mat_t = self.transpose();

        let mat_slice1 =
            unsafe { std::slice::from_raw_parts_mut(mat_t.arrays, mat_t.rows) };

        let mat_slice2 =
            unsafe { std::slice::from_raw_parts_mut(other.arrays, other.rows) };

        for i in 0..self.cols {
            result[i] = Array::zeros(other.rows);

            for j in 0..other.rows {
                result[i].set(mat_slice1[j].dotp(&mat_slice2[i]), j);
            }
        }

        Matrix {
            rows: other.rows,
            cols: self.cols,
            arrays: result.as_mut_ptr(),
        }
    }

    /// Returns the element at the index (i,j)
    ///
    /// # Arguments
    ///
    /// * `i` - the i-th row index.
    /// * `j` - the j-th column index.
    ///
    /// # Panics
    ///
    /// If any of the argument indexes go beyond the dimensions of the matrix
    /// the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix and get the value at (1,0)
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    ///
    /// assert_eq!(3.0, a.get(1, 0));
    /// // Use the `[]`-operator twice as a shorthand for indexing.
    /// assert_eq!(3.0, a[1][0]);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        slice[i].get(j)
    }

    /// Returns a subsection of a row in the matrix as an Array without modifying the matrix itself.
    ///
    /// # Arguments
    ///
    /// * `row` - the row of the matrix to splice.
    /// * `first` - the first index of the Array to splice from.
    /// * `last` - the last (exclusive) index of the Array to splice from.
    ///
    /// # Panics
    ///
    /// The first index must be strictly smaller than the last index. Also the indexes must be
    /// within the bounds of the underlying Array. Otherwise the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix and get a copy of the second row with `splice()`
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    ///
    /// assert_eq!(Array::from(&mut [3.0, 5.0]), a.splice(1, 0, 2));
    /// ```
    pub fn splice(&self, row: usize, first: usize, last: usize) -> Array {
        assert!(first < last, "ERROR - matrix splice: first index must be smaller than last index.");
        let slice = unsafe {
            let layout = Layout::array::<f64>(last - first).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut f64, last - first)
        };

        for col in first..last {
            slice[(col - first) as usize] = self.get(row, col);
        }

        Array::from(slice)
    }

    /// Changes the element at the index (i,j).
    ///
    /// # Arguments
    ///
    /// * `val` - the new value for the element to be changed.
    /// * `i` - the i-th row index.
    /// * `j` - the j-th column index.
    ///
    /// # Panics
    ///
    /// If any of the argument indexes go beyond the dimensions of the matrix
    /// the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a 2x2 matrix and set the value at (1,0) to 8.0
    /// let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
    ///
    /// a.set(8.0, 1, 0);
    /// // use the `[]`-operator twice as a shorthand for indexing
    /// // a[1][0] = 8.0;
    /// assert_eq!(8.0, a.get(1, 0));
    /// ```
    pub fn set(&mut self, val: f64, i: usize, j: usize) {
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        slice[i].set(val, j);
    }

    pub fn set_row(&mut self, arr: Array, row: usize) {
        let mut offset: usize = 0;
        if arr.len() < self.cols {
            offset = self.cols - arr.len();
        }

        for elem in 0..arr.len() {
            self.set(arr.get(elem), row, elem + offset);
        }
    }

    /// Returns the dimensions of a matrix in the form of a tuple.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns a raw mutable pointer of the elements in a matrix.
    pub fn to_raw(mat: Matrix) -> *mut Matrix {
        Box::into_raw(Box::new(mat))
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.to_string())
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows {
            return false;
        }

        if self.cols != other.cols {
            return false;
        }

        let slice1 = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };

        let slice2 = unsafe { std::slice::from_raw_parts_mut(other.arrays, other.rows) };

        for i in 0..self.rows {
            if slice1[i] != slice2[i] {
                return false;
            }
        }

        true
    }
}

impl Deref for Matrix {
    type Target = [Array];
    fn deref(&self) -> &[Array] {
        unsafe { std::slice::from_raw_parts(self.arrays, self.rows) }
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut [Array] {
        unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) }
    }
}

impl Index<usize> for Matrix {
    type Output = Array;

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < self.rows, "ERROR - Matrix: Index out of bounds.");
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };
        &slice[i]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rows, "ERROR - Matrix: Index out of bounds.");
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arrays, self.rows) };
        &mut slice[index]
    }
}

impl Add for Matrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.plus(&other)
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.minus(&other)
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.elem_mult(&other)
    }
}

impl Neg for Matrix {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scalar(-1.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn index() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);

        assert_eq!(3.0, a[1][0]);
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds_rows() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        
        a[2][1];
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds_columns() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        
        a[1][2];
    }

    #[test]
    fn zeros() {
        let z = Matrix::zeros(2, 2);
        let r = Matrix::new(&mut [Array::from(&mut [0.0, 0.0]), Array::from(&mut [0.0, 0.0])]);

        assert_eq!(r, z);
    }

    #[test]
    fn ones() {
        let o = Matrix::ones(2, 2);
        let r = Matrix::new(&mut [Array::from(&mut [1.0, 1.0]), Array::from(&mut [1.0, 1.0])]);

        assert_eq!(r, o);
    }

    #[test]
    fn identity() {
        let i = Matrix::identity(2);
        let r = Matrix::new(&mut [Array::from(&mut [1.0, 0.0]), Array::from(&mut [0.0, 1.0])]);

        assert_eq!(r, i);
    }

    #[test]
    fn add() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [3.0, 5.0]), Array::from(&mut [8.0, 13.0])]);
        assert_eq!(r, a + b);
    }

    #[test]
    fn sub() {
        let a = Matrix::new(&mut [Array::from(&mut [3.0, 5.0]), Array::from(&mut [8.0, 13.0])]);
        let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        assert_eq!(r, a - b);
    }

    #[test]
    fn scalar() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [2.0, 4.0]), Array::from(&mut [6.0, 10.0])]);

        assert_eq!(r, a.scalar(2.0));
    }

    #[test]
    fn neg() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 5.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [-1.0, -2.0]), Array::from(&mut [-3.0, -5.0])]);

        assert_eq!(r, -a);
    }

    #[test]
    fn elem_mult() {
        let a = Matrix::new(&mut [Array::from(&mut [3.0, 5.0]), Array::from(&mut [8.0, 13.0])]);
        let b = Matrix::new(&mut [Array::from(&mut [2.0, 3.0]), Array::from(&mut [5.0, 8.0])]);
        let r = Matrix::new(&mut [
            Array::from(&mut [6.0, 15.0]),
            Array::from(&mut [40.0, 104.0]),
        ]);
        assert_eq!(r, a * b);
    }

    #[test]
    fn mult() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 4.0])]);
        let r = Matrix::new(&mut [
            Array::from(&mut [7.0, 10.0]),
            Array::from(&mut [15.0, 22.0]),
        ]);
        assert_eq!(r, a.mult(&a));
    }

    #[test]
    fn transpose() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 4.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [1.0, 3.0]), Array::from(&mut [2.0, 4.0])]);
        assert_eq!(r, a.transpose());
    }

    #[test]
    fn get() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 4.0])]);
        assert_eq!(3.0, a.get(1, 0));
    }

    #[test]
    fn set() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 4.0])]);
        let r = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 8.0])]);

        a.set(8.0, 1, 1);

        assert_eq!(r, a);
    }

    #[test]
    fn iterator() {
        let a = Matrix::new(&mut [Array::from(&mut [1.0, 2.0]), Array::from(&mut [3.0, 4.0])]);
        let first = Array::from(&mut [1.0, 2.0]);
        let second = Array::from(&mut [3.0, 4.0]);

        let mut it = a.iter();

        assert_eq!(it.next(), Some(first).as_ref());
        assert_eq!(it.next(), Some(second).as_ref());
    }
}
