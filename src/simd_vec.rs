use rayon::prelude::*;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};
use std::ops::{Index, IndexMut};
use std::simd::SimdElement;

use crate::chunk::SimdChunk;

#[derive(Debug)]
pub struct SimdVec<T: SimdElement> {
    data: Vec<SimdChunk<T>>,
    chunk_counts: [usize; 7],
}

impl<T: SimdElement> SimdVec<T> {
    pub fn len(&self) -> usize {
        64 * self.chunk_counts[0]
            + 32 * self.chunk_counts[1]
            + 16 * self.chunk_counts[2]
            + 8 * self.chunk_counts[3]
            + 4 * self.chunk_counts[4]
            + 2 * self.chunk_counts[5]
            + self.chunk_counts[6]
    }

    pub fn into_vec(self) -> Vec<T>
    where
        T: From<i8>,
    {
        let mut result = vec![T::from(0); self.len()];
        let mut start = 0;

        for chunk in self.data {
            let size = chunk.len();
            chunk.copy_to_slice(&mut result[start..start + size]);
            start += size;
        }

        result
    }
}

impl<T: SimdElement> Index<usize> for SimdVec<T> {
    type Output = T;

    fn index(&self, mut index: usize) -> &T {
        let mut offset = 0;
        // 64
        if index < self.chunk_counts[0] * 64 {
            return self.data[offset + index / 64].get(index % 64);
        }
        (offset, index) = (
            offset + self.chunk_counts[0],
            index - self.chunk_counts[0] * 64,
        );
        // 32
        if index < self.chunk_counts[1] * 32 {
            return self.data[offset].get(index % 32);
        }
        (offset, index) = (
            offset + self.chunk_counts[1],
            index - self.chunk_counts[1] * 32,
        );
        // 16
        if index < self.chunk_counts[2] * 16 {
            return self.data[offset].get(index % 16);
        }
        (offset, index) = (
            offset + self.chunk_counts[2],
            index - self.chunk_counts[2] * 16,
        );
        // 8
        if index < self.chunk_counts[3] * 8 {
            return self.data[offset].get(index % 8);
        }
        (offset, index) = (
            offset + self.chunk_counts[3],
            index - self.chunk_counts[3] * 8,
        );
        // 4
        if index < self.chunk_counts[4] * 4 {
            return self.data[offset].get(index % 4);
        }
        (offset, index) = (
            offset + self.chunk_counts[4],
            index - self.chunk_counts[4] * 4,
        );
        // 2
        if index < self.chunk_counts[5] * 2 {
            return self.data[offset].get(index % 2);
        }
        (offset, index) = (
            offset + self.chunk_counts[5],
            index - self.chunk_counts[5] * 2,
        );
        // 1
        self.data[offset].get(index)
    }
}

impl<T: SimdElement> IndexMut<usize> for SimdVec<T> {
    fn index_mut(&mut self, mut index: usize) -> &mut T {
        let mut offset = 0;
        // 64
        if index < self.chunk_counts[0] * 64 {
            return self.data[offset + index / 64].get_mut(index % 64);
        }
        (offset, index) = (
            offset + self.chunk_counts[0],
            index - self.chunk_counts[0] * 64,
        );
        // 32
        if index < self.chunk_counts[1] * 32 {
            return self.data[offset].get_mut(index % 32);
        }
        (offset, index) = (
            offset + self.chunk_counts[1],
            index - self.chunk_counts[1] * 32,
        );
        // 16
        if index < self.chunk_counts[2] * 16 {
            return self.data[offset].get_mut(index % 16);
        }
        (offset, index) = (
            offset + self.chunk_counts[2],
            index - self.chunk_counts[2] * 16,
        );
        // 8
        if index < self.chunk_counts[3] * 8 {
            return self.data[offset].get_mut(index % 8);
        }
        (offset, index) = (
            offset + self.chunk_counts[3],
            index - self.chunk_counts[3] * 8,
        );
        // 4
        if index < self.chunk_counts[4] * 4 {
            return self.data[offset].get_mut(index % 4);
        }
        (offset, index) = (
            offset + self.chunk_counts[4],
            index - self.chunk_counts[4] * 4,
        );
        // 2
        if index < self.chunk_counts[5] * 2 {
            return self.data[offset].get_mut(index % 2);
        }
        (offset, index) = (
            offset + self.chunk_counts[5],
            index - self.chunk_counts[5] * 2,
        );
        // 1
        self.data[offset].get_mut(index)
    }
}

impl<T: SimdElement + Display> Display for SimdVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.data
                .iter()
                .flat_map(|x| x.as_array())
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl<T: SimdElement> From<&[T]> for SimdVec<T> {
    fn from(value: &[T]) -> Self {
        // Current SIMD width to pack
        let mut lanes = 64;
        // Current index into `value`
        let mut i = 0;
        // Current index into `chunk_counts`
        let mut j = 0;
        // Number of `SimdChunk`s of each width (from 64 to 1)
        let mut chunk_counts = [0, 0, 0, 0, 0, 0, 0];
        let mut data = vec![];

        // Greedily pack values into `SimdChunk`s
        while i < value.len() {
            while i + lanes > value.len() {
                lanes /= 2;
                j += 1;
            }
            data.push(SimdChunk::from(&value[i..i + lanes]));
            chunk_counts[j] += 1;
            i += lanes;
        }

        SimdVec { data, chunk_counts }
    }
}

impl<T: SimdElement> From<Vec<T>> for SimdVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self::from(&value[..])
    }
}

impl<T: SimdElement> From<&Vec<T>> for SimdVec<T> {
    fn from(value: &Vec<T>) -> Self {
        Self::from(&value[..])
    }
}

macro_rules! impl_trait_for_type {
    ($trait:ident, $op:tt, $method:ident, $simd_type:ty) => {
        impl $trait for &SimdVec<$simd_type> {
            type Output = SimdVec<$simd_type>;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "SimdVecs must have same size."
                );
                SimdVec {
                    data: self
                        .data
                        .par_iter()
                        .zip(&rhs.data)
                        .map(|(x, y)| x $op y)
                        .collect(),
                    chunk_counts: self.chunk_counts.clone(),
                }
            }
        }

        impl $trait for SimdVec<$simd_type> {
            type Output = SimdVec<$simd_type>;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "SimdVecs must have same size."
                );
                SimdVec {
                    data: self
                        .data
                        .par_iter()
                        .zip(&rhs.data)
                        .map(|(x, y)| x $op y)
                        .collect(),
                    chunk_counts: self.chunk_counts.clone(),
                }
            }
        }
    };
}

macro_rules! impl_simdvec_op {
    ($trait:ident, $op:tt, $method:ident) => {
        impl_trait_for_type!($trait, $op, $method, f32);
        impl_trait_for_type!($trait, $op, $method, f64);
        impl_trait_for_type!($trait, $op, $method, i8);
        impl_trait_for_type!($trait, $op, $method, i16);
        impl_trait_for_type!($trait, $op, $method, i32);
        impl_trait_for_type!($trait, $op, $method, i64);
    };
}

macro_rules! impl_scalar_product_for_type {
    ($simd_type:ty) => {
        impl Mul<$simd_type> for &SimdVec<$simd_type> {
            type Output = SimdVec<$simd_type>;

            fn mul(self, rhs: $simd_type) -> Self::Output {
                SimdVec {
                    data: self.data.par_iter().map(|x| x * rhs).collect(),
                    chunk_counts: self.chunk_counts.clone(),
                }
            }
        }

        impl Mul<$simd_type> for SimdVec<$simd_type> {
            type Output = SimdVec<$simd_type>;

            fn mul(self, rhs: $simd_type) -> Self::Output {
                SimdVec {
                    data: self.data.par_iter().map(|x| x * rhs).collect(),
                    chunk_counts: self.chunk_counts.clone(),
                }
            }
        }
    };
}

impl<T: SimdElement> Sum for SimdVec<T>
where
    SimdVec<T>: Add<Output = SimdVec<T>>,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut result = iter.next().unwrap();
        while let Some(simd_vec) = iter.next() {
            result = result + simd_vec;
        }
        result
    }
}

// Element-wise vector ops
// =================================
// Element-wise vector addition
impl_simdvec_op!(Add, +, add);
// Element-wise vector subtraction
impl_simdvec_op!(Sub, -, sub);
// Element-wise vector product
impl_simdvec_op!(Mul, *, mul);

// Scalar products
// =================================
impl_scalar_product_for_type!(f32);
impl_scalar_product_for_type!(f64);
impl_scalar_product_for_type!(i8);
impl_scalar_product_for_type!(i16);
impl_scalar_product_for_type!(i32);
impl_scalar_product_for_type!(i64);
