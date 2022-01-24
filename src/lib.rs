//! `moonalloy` - A Safe Scientific Computing Library for the 21st-century
//!
//! `moonalloy` provides a library structures and implementations for techniques used in the
//! domain of scientific computing/computational science, such as tool for numerical
//! linear algebra. The library leverages rust's immutability by-default to achieve purity for
//! the mathematical functions.
//!
//! All the structures and functions are accessible in other languages through the FFI for the C
//! ABI. Lua (Luajit) is a first-class supported language. For frontend wrappers for Lua see
//! ![moonalloy-luajit](https://git.hacktheoxidation.xyz/HackTheOxidation/moonalloy-luajit).

pub mod linalg;

use crate::linalg::array::Array;
use crate::linalg::matrix::Matrix;
use crate::linalg::methods::gauss_elimination;
use std::ffi::CString;
use std::os::raw::c_char;

/// FFI-function for calculating the sum of the elements of an array.
#[no_mangle]
pub extern "C" fn array_sum(ptr: *mut Array) -> f64 {
    let arr = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    arr.sum()
}

/// FFI-function that prints the contents of an array to stdout.
#[no_mangle]
pub extern "C" fn array_print(ptr: *mut Array) {
    let arr = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    println!("{}", arr);
}

/// FFI-function that creates a new, empty array.
#[no_mangle]
pub extern "C" fn array_new() -> *mut Array {
    Box::into_raw(Box::new(Array::new()))
}

/// FFI-function that multiplies the array with a scalar value.
#[no_mangle]
pub extern "C" fn array_scalar(ptr: *mut Array, scal: f64) -> *mut Array {
    let arr = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };

    Array::to_raw(arr.scalar(scal))
}

/// FFI-function that adds two arrays together and returns a copy of the result.
#[no_mangle]
pub extern "C" fn array_add(ptr1: *const Array, ptr2: *const Array) -> *mut Array {
    let arr1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let arr2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Array::to_raw(arr1.plus(arr2))
}

/// FFI-function that subtracts two arrays and returns a copy of the result.
#[no_mangle]
pub extern "C" fn array_sub(ptr1: *const Array, ptr2: *const Array) -> *mut Array {
    let arr1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let arr2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    let result = arr1.minus(arr2);

    Array::to_raw(result)
}

/// FFI-function that multiplies two arrays elements by elements and returns a copy of the result.
#[no_mangle]
pub extern "C" fn array_mult(ptr1: *const Array, ptr2: *const Array) -> *mut Array {
    let arr1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let arr2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    let result = arr1.mult(arr2);

    Array::to_raw(result)
}

/// FFI-function that takes the dot product of two Arrays and returns a copy of the result.
#[no_mangle]
pub extern "C" fn array_dotp(ptr1: *const Array, ptr2: *const Array) -> f64 {
    let arr1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let arr2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    arr1.dotp(arr2)
}

/// FFI-function that returns a string representation of the contents of an Array.
#[no_mangle]
pub extern "C" fn array_to_string(ptr: *const Array) -> *const c_char {
    let arr = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };

    let c_str = CString::new(arr.to_string().as_str()).unwrap();
    let result = c_str.as_ptr();
    std::mem::forget(c_str);
    result
}

/// FFI-function that returns a concatenation of two Arrays.
#[no_mangle]
pub extern "C" fn array_concat(ptr1: *const Array, ptr2: *const Array) -> *mut Array {
    let arr1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let arr2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    let result = arr1.concat(arr2);

    Array::to_raw(result)
}

/// FFI-function that returns an Array of zeros.
#[no_mangle]
pub extern "C" fn array_zeros(len: i32) -> *mut Array {
    let array = Array::zeros(len as usize);
    Array::to_raw(array)
}

/// FFI-function that returns an Array of ones.
#[no_mangle]
pub extern "C" fn array_ones(len: i32) -> *mut Array {
    let array = Array::ones(len as usize);
    Array::to_raw(array)
}

// Matrix
/// FFI-function that returns a `n`x`m` matrix of zeros.
#[no_mangle]
pub extern "C" fn matrix_zeros(rows: i32, cols: i32) -> *mut Matrix {
    let mat = Matrix::zeros(rows as usize, cols as usize);
    Matrix::to_raw(mat)
}

/// FFI-function that returns a `n`x`m` matrix of ones.
#[no_mangle]
pub extern "C" fn matrix_ones(rows: i32, cols: i32) -> *mut Matrix {
    let mat = Matrix::ones(rows as usize, cols as usize);
    Matrix::to_raw(mat)
}

/// FFI-function that returns an identity matrix of rank n.
#[no_mangle]
pub extern "C" fn matrix_identity(len: i32) -> *mut Matrix {
    let mat = Matrix::identity(len as usize);
    Matrix::to_raw(mat)
}

/// FFI-function that prints the contents of a matrix to stdout.
#[no_mangle]
pub extern "C" fn matrix_print(ptr: *mut Matrix) {
    let mat = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    println!("{}", mat);
}

/// FFI-function that returns a string representation of a matrix.
#[no_mangle]
pub extern "C" fn matrix_to_string(ptr: *const Matrix) -> *const c_char {
    let mat = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };

    let c_str = CString::new(mat.to_string().as_str()).unwrap();
    let result = c_str.as_ptr();
    std::mem::forget(c_str);
    result
}

/// FFI-function that adds two matrices together and returns a copy of the result.
#[no_mangle]
pub extern "C" fn matrix_add(ptr1: *const Matrix, ptr2: *const Matrix) -> *mut Matrix {
    let mat1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let mat2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Matrix::to_raw(mat1.plus(mat2))
}

/// FFI-function that multiplies the elements of a matrix with a scalar value.
#[no_mangle]
pub extern "C" fn matrix_scalar(ptr: *const Matrix, scal: f64) -> *mut Matrix {
    let mat = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };

    Matrix::to_raw(mat.scalar(scal))
}

/// FFI-function that subtracts two matrices and returns a copy of the result.
#[no_mangle]
pub extern "C" fn matrix_sub(ptr1: *const Matrix, ptr2: *const Matrix) -> *mut Matrix {
    let mat1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let mat2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Matrix::to_raw(mat1.minus(mat2))
}

/// FFI-function that performs element-wise multiplication of two matrices.
#[no_mangle]
pub extern "C" fn matrix_elem_mult(ptr1: *const Matrix, ptr2: *const Matrix) -> *mut Matrix {
    let mat1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let mat2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Matrix::to_raw(mat1.elem_mult(mat2))
}

/// FFI-function that transposes a matrix and returns the result as a copy.
#[no_mangle]
pub extern "C" fn matrix_transpose(ptr: *const Matrix) -> *mut Matrix {
    let mat = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };

    Matrix::to_raw(mat.transpose())
}

/// FFI-function that multiplies two matrices and returns a copy of the result.
#[no_mangle]
pub extern "C" fn matrix_mult(ptr1: *const Matrix, ptr2: *const Matrix) -> *mut Matrix {
    let mat1 = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let mat2 = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Matrix::to_raw(mat1.mult(mat2))
}

/// FFI-function that solves a system of linear equations with Gauss Elimination.
#[no_mangle]
pub extern "C" fn linalg_gauss(ptr1: *const Matrix, ptr2: *const Array) -> *mut Array {
    let a = unsafe {
        assert!(!ptr1.is_null());
        &*ptr1
    };

    let b = unsafe {
        assert!(!ptr2.is_null());
        &*ptr2
    };

    Array::to_raw(gauss_elimination(a.clone(), b.clone()))
}
