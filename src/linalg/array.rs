//! Array - An implementation of a mathematical vector/array
//!
//! This module contains structures and functions for manipulating vectors/arrays in Linear
//! Algebra.

use std::alloc::{alloc, Layout};
use std::fmt::*;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};

/// A representation of a mathematical array/vector
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Array {
    /// Number of elements in the Array
    len: usize,
    /// Elements of the Array, stored as a mutable pointer
    arr: *mut f64,
}

impl Array {
    /// Returns a new Array with no elements
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new empty Array
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::new();
    /// ```
    pub fn new() -> Array {
        let arr_slice = unsafe {
            let layout = Layout::new::<f64>();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut f64, 0)
        };

        Array {
            len: 0,
            arr: arr_slice.as_mut_ptr(),
        }
    }

    /// Creates a new Array from a slice of elements
    ///
    /// # Arguments
    ///
    /// * `slice` - A mutable slice of float values. This will become the internal values of the
    /// Array.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    /// ```
    pub fn from(slice: &mut [f64]) -> Array {
        Array {
            len: slice.len(),
            arr: slice.as_mut_ptr(),
        }
    }

    /// Calculate the sum of all the elements in the Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(6.0, array.sum());
    /// ```
    pub fn sum(&self) -> f64 {
        let mut s: f64 = 0.0;
        let v = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            s += v[i];
        }
        s
    }

    /// Calculate the average of all the elements in the Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(2.0, array.average());
    /// ```
    pub fn average(&self) -> f64 {
        let mut s: f64 = 0.0;
        let v = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            s += v[i];
        }
        s / self.len as f64
    }

    /// Calculate the norm of the Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 3.0 and 4.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [3.0, 4.0]);
    ///
    /// assert_eq!(5.0, array.norm());
    /// ```
    pub fn norm(&self) -> f64 {
        let mut n: f64 = 0.0;
        let v = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            n += v[i] * v[i];
        }

        n.sqrt()
    }

    /// Add a scalar value to every element in the Array
    ///
    /// # Arguments
    ///
    /// * `scalar` - scalar value to add.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [3.0, 4.0, 5.0]), array.scalar_add(2.0));
    /// ```
    pub fn scalar_add(&self, scalar: f64) -> Array {
        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr_slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            result[i] = scalar + arr_slice[i];
        }

        Array {
            arr: result.as_mut_ptr(),
            len: self.len,
        }
    }

    /// Subtract a scalar value from every element in the Array
    ///
    /// # Arguments
    ///
    /// * `scalar` - scalar value to subtract.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [-1.0, 0.0, 1.0]), array.scalar_sub(2.0));
    /// ```
    pub fn scalar_sub(&self, scalar: f64) -> Array {
        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr_slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            result[i] = arr_slice[i] - scalar;
        }

        Array {
            arr: result.as_mut_ptr(),
            len: self.len,
        }
    }

    /// Multiply every element in the Array with a scalar value
    ///
    /// # Arguments
    ///
    /// * `scalar` - scalar value to multiply with.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a new Array containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [2.0, 4.0, 6.0]), array.scalar_mult(2.0));
    /// ```
    pub fn scalar_mult(&self, scalar: f64) -> Array {
        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr_slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        for i in 0..self.len() {
            result[i] = scalar * arr_slice[i];
        }

        Array {
            arr: result.as_mut_ptr(),
            len: self.len,
        }
    }

    /// Add two Arrays without modifying either Array.
    ///
    /// # Arguments
    ///
    /// * `other` - the other Array to add
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    /// let b = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [2.0, 4.0, 6.0]), a.plus(&b));
    /// // You can use the `+`-operator as a shorthand for this
    /// assert_eq!(Array::from(&mut [2.0, 4.0, 6.0]), a + b);
    /// ```
    pub fn plus(&self, other: &Array) -> Array {
        assert_eq!(self.len(), other.len(), "Lengths are different!");

        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr1 = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        let arr2 = unsafe { std::slice::from_raw_parts_mut(other.arr, other.len()) };

        for i in 0..self.len() {
            result[i] = arr1[i] + arr2[i];
        }

        Array {
            len: result.len(),
            arr: result.as_mut_ptr(),
        }
    }

    /// Performs substraction on two Arrays without modifying either Array.
    ///
    /// # Arguments
    ///
    /// * `other` - the other Array to substract
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    /// let b = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [0.0, 0.0, 0.0]), a.minus(&b));
    /// // You can use the `-`-operator as a shorthand for this
    /// assert_eq!(Array::from(&mut [0.0, 0.0, 0.0]), a - b);
    /// ```
    pub fn minus(&self, other: &Array) -> Array {
        assert_eq!(self.len(), other.len(), "Lengths are different!");

        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr1 = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        let arr2 = unsafe { std::slice::from_raw_parts_mut(other.arr, other.len()) };

        for i in 0..self.len() {
            result[i] = arr1[i] - arr2[i];
        }

        Array {
            len: result.len(),
            arr: result.as_mut_ptr(),
        }
    }

    /// Performs multiplication on two Arrays without modifying either Array.
    ///
    /// # Arguments
    ///
    /// * `other` - the other Array to multiply with
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    /// let b = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(Array::from(&mut [1.0, 4.0, 9.0]), a.mult(&b));
    /// // You can use the `*`-operator as a shorthand for this
    /// assert_eq!(Array::from(&mut [1.0, 4.0, 9.0]), a * b);
    /// ```
    pub fn mult(&self, other: &Array) -> Array {
        assert_eq!(self.len(), other.len(), "Lengths are different!");

        let result = unsafe {
            let layout = Layout::array::<f64>(self.len()).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.len())
        };

        let arr1 = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        let arr2 = unsafe { std::slice::from_raw_parts_mut(other.arr, other.len()) };

        for i in 0..self.len() {
            result[i] = arr1[i] * arr2[i];
        }

        Array {
            len: result.len(),
            arr: result.as_mut_ptr(),
        }
    }

    /// Calculates the dot product on two Arrays without modifying either Array.
    /// Returns a single floating-point value.
    ///
    /// # Arguments
    ///
    /// * `other` - the other Array calculate the dot product with
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    /// let b = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(14.0, a.dotp(&b));
    /// ```
    pub fn dotp(&self, other: &Array) -> f64 {
        let arr = self.mult(other);
        let v = unsafe { std::slice::from_raw_parts_mut(arr.arr, arr.len()) };
        v.iter().sum()
    }

    /// Concatenate with another Array. This will modify the original array.
    ///
    /// # Arguments
    ///
    /// * `other` - the other Array calculate the dot product with
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    /// let b = Array::from(&mut [4.0, 5.0]);
    /// a.concat(&b);
    ///
    /// assert_eq!(Array::from(&mut [1.0, 2.0, 3.0, 4.0, 5.0]), a.dotp(&b));
    /// ```
    pub fn concat(&self, other: &Array) -> Array {
        let len = self.len() + other.len();
        let result = unsafe {
            let layout = Layout::array::<f64>(len).unwrap();
            let ptr = alloc(layout) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, len)
        };

        let arr1 = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        let arr2 = unsafe { std::slice::from_raw_parts_mut(other.arr, other.len()) };

        let mut i = 0;
        for elem in arr1.iter() {
            result[i] = *elem;
            i += 1;
        }

        for elem in arr2.iter() {
            result[i] = *elem;
            i += 1;
        }

        Array {
            len: result.len(),
            arr: result.as_mut_ptr(),
        }
    }

    /// Returns a string representation of the Array.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// // Creates two new Arrays both containing the values 1.0, 2.0 and 3.0
    /// use moonalloy::linalg::array::Array;
    /// let a = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// println!("{}", a.to_string());
    /// // The to_string() is not necessary since Array implements the `Display` trait.
    /// println!("{}", a);
    /// ```
    pub fn to_string(&self) -> String {
        let array_slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        format!("Array: {:?}", array_slice)
    }

    /// Returns a raw mutable pointer to the Array.
    /// This is useful for FFI purposes.
    ///
    /// # Arguments
    ///
    /// * `arr` - the Array to be converted into a raw pointer
    pub fn to_raw(arr: Array) -> *mut Array {
        Box::into_raw(Box::new(arr))
    }

    /// Creates a new Array of length `len` all where all elements have the value of `val`.
    ///
    /// # Arguments
    ///
    /// * `val` - the value for all elements in the new Array
    /// * `len` - the number of elements in the new Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an Array with 3 elements, where all have the value of 2.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::of(2.0, 3);
    ///
    /// assert_eq!(Array::from(&mut [2.0, 2.0, 2.0]), array);
    /// ```
    pub fn of(val: f64, len: usize) -> Array {
        let arr_slice = unsafe {
            let layout = Layout::array::<f64>(len).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut f64, len)
        };

        for i in 0..len {
            arr_slice[i] = val;
        }

        Array {
            arr: arr_slice.as_mut_ptr(),
            len,
        }
    }

    /// Creates a new Array of length `len` all where all elements are set to 0.0.
    ///
    /// # Arguments
    ///
    /// * `len` - the number of elements in the new Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an Array with 3 elements, where all have the value of 0.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::zeros(3);
    ///
    /// assert_eq!(Array::from(&mut [0.0, 0.0, 0.0]), array);
    /// ```
    pub fn zeros(len: usize) -> Array {
        Array::of(0.0, len)
    }

    /// Creates a new Array of length `len` all where all elements are set to 1.0.
    ///
    /// # Arguments
    ///
    /// * `len` - the number of elements in the new Array
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an Array with 3 elements, where all have the value of 1.0
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::ones(3);
    ///
    /// assert_eq!(Array::from(&mut [1.0, 1.0, 1.0]), array);
    /// ```
    pub fn ones(len: usize) -> Array {
        Array::of(1.0, len)
    }

    /// Returns the value at index: `index` in the Array.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of the requested value.
    ///
    /// # Panics
    ///
    /// The `index` must be smaller than the length of the Array otherwise the code will panic with
    /// an index out of bounds error.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an Array with 3 elements
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(2.0, array.get(1));
    /// // The shorthand for this is the `[]`-operator
    /// assert_eq!(2.0, array[1]);
    /// ```
    pub fn get(&self, index: usize) -> f64 {
        assert!(
            index < self.len(),
            "ERROR - Array get: Index out of bounds."
        );
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        slice[index]
    }

    /// Mutates the value at index: `index` in the Array.
    ///
    /// # Arguments
    ///
    /// * `val` - the new value to be set.
    /// * `index` - the index of the requested value.
    ///
    /// # Panics
    ///
    /// The `index` must be smaller than the length of the Array otherwise the code will panic with
    /// an index out of bounds error.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an Array with 3 elements
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0]);
    ///
    /// array.set(5.0, 1);
    /// // use the `[]`-operator as a shorthand
    /// // array[1] = 5.0;
    /// assert_eq!(5.0, array[1]);
    /// ```
    pub fn set(&mut self, val: f64, index: usize) {
        assert!(
            index < self.len(),
            "ERROR - Array get: Index out of bounds."
        );
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        slice[index] = val;
    }

    /// Returns a copy of a section of the Array
    ///
    /// # Arguments
    ///
    /// * `first` - first index of the Array to copy from.
    /// * `last` - last index (exclusive) of the Array to copy from.
    ///
    /// # Panics
    ///
    /// `first` must be strictly smaller than `last` otherwise the code will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use moonalloy::linalg::array::Array;
    /// let array = Array::from(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);
    ///
    /// assert_eq(Array::from(&mut [2.0, 3.0]), array.splice(1, 3));
    /// ```
    pub fn splice(&self, first: usize, last: usize) -> Array {
        assert!(
            first < last,
            "ERROR - Array splice: first index must be before last index"
        );
        let arr_slice = unsafe {
            let layout = Layout::array::<f64>(last - first).unwrap();
            let ptr = alloc(layout);
            std::slice::from_raw_parts_mut(ptr as *mut f64, last - first)
        };

        for i in first..last {
            arr_slice[i - first] = self.get(i);
        }

        Array::from(arr_slice)
    }

    /// Returns the contents of the Array as a slice of floating-point values.
    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) }
    }

    /// Returns the number of elements in the Array
    pub fn len(&self) -> usize {
        self.len
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.to_string())
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let slice1 = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };

        let slice2 = unsafe { std::slice::from_raw_parts_mut(other.arr, other.len()) };

        for i in 0..self.len() {
            if slice1[i] != slice2[i] {
                return false;
            }
        }

        true
    }
}

impl Deref for Array {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.arr, self.len()) }
    }
}

impl DerefMut for Array {
    fn deref_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) }
    }
}

impl Index<usize> for Array {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < self.len(), "ERROR - Array: Index out of bounds.");
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };
        &slice[i]
    }
}

impl IndexMut<usize> for Array {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < self.len(),
            "ERROR - Array get: Index out of bounds."
        );
        let slice = unsafe { std::slice::from_raw_parts_mut(self.arr, self.len()) };
        &mut slice[index]
    }
}

impl Add for Array {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.plus(&other)
    }
}

impl Sub for Array {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.minus(&other)
    }
}

impl Mul for Array {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mult(&other)
    }
}

impl Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.scalar_mult(-1.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new() {
        let n = Array::new();
        let f = Array::from(&mut []);

        assert_eq!(n, f);
    }

    #[test]
    fn index() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);

        assert_eq!(2.0, a[1]);
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);

        a[3];
    }

    #[test]
    fn sum() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);

        assert_eq!(6.0, a.sum());
    }

    #[test]
    fn scalar_mult() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let r = Array::from(&mut [2.0, 4.0, 6.0]);

        assert_eq!(r, a.scalar_mult(2.0))
    }

    #[test]
    fn neg() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let r = Array::from(&mut [-1.0, -2.0, -3.0]);

        assert_eq!(r, -a)
    }

    #[test]
    fn add() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let b = Array::from(&mut [2.0, 3.0, 5.0]);
        let r = Array::from(&mut [3.0, 5.0, 8.0]);

        assert_eq!(r, a + b);
    }

    #[test]
    fn sub() {
        let a = Array::from(&mut [2.0, 3.0, 5.0]);
        let b = Array::from(&mut [1.0, 2.0, 3.0]);
        let r = Array::from(&mut [1.0, 1.0, 2.0]);

        assert_eq!(r, a - b);
    }

    #[test]
    fn mult() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let b = Array::from(&mut [2.0, 3.0, 5.0]);
        let r = Array::from(&mut [2.0, 6.0, 15.0]);

        assert_eq!(r, a * b);
    }

    #[test]
    fn dotp() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let b = Array::from(&mut [2.0, 3.0, 5.0]);

        assert_eq!(23.0, a.dotp(&b));
    }

    #[test]
    fn concat() {
        let a = Array::from(&mut [1.0, 2.0]);
        let b = Array::from(&mut [3.0, 5.0]);
        let r = Array::from(&mut [1.0, 2.0, 3.0, 5.0]);

        assert_eq!(r, a.concat(&b));
    }

    #[test]
    fn zeros() {
        let a = Array::zeros(3);
        let r = Array::from(&mut [0.0, 0.0, 0.0]);

        assert_eq!(r, a);
    }

    #[test]
    fn ones() {
        let a = Array::ones(3);
        let r = Array::from(&mut [1.0, 1.0, 1.0]);

        assert_eq!(r, a);
    }

    #[test]
    fn get() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);

        assert_eq!(2.0, a.get(1));
    }

    #[test]
    fn set() {
        let mut a = Array::from(&mut [1.0, 2.0, 3.0]);
        let r = Array::from(&mut [5.0, 2.0, 3.0]);

        a.set(5.0, 0);

        assert_eq!(r, a);
    }

    #[test]
    fn iterator() {
        let a = Array::from(&mut [1.0, 2.0, 3.0]);
        let mut it = a.iter();

        assert_eq!(*it.next().unwrap(), 1.0_f64);
        assert_eq!(*it.next().unwrap(), 2.0_f64);
        assert_eq!(*it.next().unwrap(), 3.0_f64);
    }

    #[test]
    fn splice() {
        let expected = Array::from(&mut [2.0, 3.0]);
        let a = Array::from(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);

        let actual = a.splice(1, 3);
        assert_eq!(expected, actual);
    }
}
