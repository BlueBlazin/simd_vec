use std::ops::{Add, Mul, Sub};
use std::simd::{Simd, SimdElement};

#[derive(Debug, PartialEq)]
pub enum SimdChunk<T: SimdElement> {
    Width1(Simd<T, 1>),
    Width2(Simd<T, 2>),
    Width4(Simd<T, 4>),
    Width8(Simd<T, 8>),
    Width16(Simd<T, 16>),
    Width32(Simd<T, 32>),
    Width64(Simd<T, 64>),
}

impl<T: SimdElement> SimdChunk<T> {
    pub fn as_array(&self) -> &[T] {
        match self {
            SimdChunk::Width1(x) => x.as_array(),
            SimdChunk::Width2(x) => x.as_array(),
            SimdChunk::Width4(x) => x.as_array(),
            SimdChunk::Width8(x) => x.as_array(),
            SimdChunk::Width16(x) => x.as_array(),
            SimdChunk::Width32(x) => x.as_array(),
            SimdChunk::Width64(x) => x.as_array(),
        }
    }

    pub fn copy_to_slice(&self, dest: &mut [T]) {
        match self {
            SimdChunk::Width1(x) => x.copy_to_slice(dest),
            SimdChunk::Width2(x) => x.copy_to_slice(dest),
            SimdChunk::Width4(x) => x.copy_to_slice(dest),
            SimdChunk::Width8(x) => x.copy_to_slice(dest),
            SimdChunk::Width16(x) => x.copy_to_slice(dest),
            SimdChunk::Width32(x) => x.copy_to_slice(dest),
            SimdChunk::Width64(x) => x.copy_to_slice(dest),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SimdChunk::Width1(_) => 1,
            SimdChunk::Width2(_) => 2,
            SimdChunk::Width4(_) => 4,
            SimdChunk::Width8(_) => 8,
            SimdChunk::Width16(_) => 16,
            SimdChunk::Width32(_) => 32,
            SimdChunk::Width64(_) => 64,
        }
    }

    pub fn get(&self, index: usize) -> &T {
        match self {
            SimdChunk::Width1(x) => &x[index],
            SimdChunk::Width2(x) => &x[index],
            SimdChunk::Width4(x) => &x[index],
            SimdChunk::Width8(x) => &x[index],
            SimdChunk::Width16(x) => &x[index],
            SimdChunk::Width32(x) => &x[index],
            SimdChunk::Width64(x) => &x[index],
        }
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        match self {
            SimdChunk::Width1(x) => &mut x[index],
            SimdChunk::Width2(x) => &mut x[index],
            SimdChunk::Width4(x) => &mut x[index],
            SimdChunk::Width8(x) => &mut x[index],
            SimdChunk::Width16(x) => &mut x[index],
            SimdChunk::Width32(x) => &mut x[index],
            SimdChunk::Width64(x) => &mut x[index],
        }
    }
}

impl<T: SimdElement> From<&[T]> for SimdChunk<T> {
    fn from(value: &[T]) -> Self {
        match value.len() {
            1 => SimdChunk::Width1(Simd::from_slice(value)),
            2 => SimdChunk::Width2(Simd::from_slice(value)),
            4 => SimdChunk::Width4(Simd::from_slice(value)),
            8 => SimdChunk::Width8(Simd::from_slice(value)),
            16 => SimdChunk::Width16(Simd::from_slice(value)),
            32 => SimdChunk::Width32(Simd::from_slice(value)),
            64 => SimdChunk::Width64(Simd::from_slice(value)),
            _ => unreachable!("Value must have length 1, 2, 4, 8, 16, 32, or 64."),
        }
    }
}

macro_rules! impl_trait_for_type {
    ($trait:ident, $op:tt, $method:ident, $simd_type:ty) => {
        impl $trait for &SimdChunk<$simd_type> {
            type Output = SimdChunk<$simd_type>;

            fn $method(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (SimdChunk::Width1(x), SimdChunk::Width1(y)) => SimdChunk::Width1(x $op y),
                    (SimdChunk::Width2(x), SimdChunk::Width2(y)) => SimdChunk::Width2(x $op y),
                    (SimdChunk::Width4(x), SimdChunk::Width4(y)) => SimdChunk::Width4(x $op y),
                    (SimdChunk::Width8(x), SimdChunk::Width8(y)) => SimdChunk::Width8(x $op y),
                    (SimdChunk::Width16(x), SimdChunk::Width16(y)) => SimdChunk::Width16(x $op y),
                    (SimdChunk::Width32(x), SimdChunk::Width32(y)) => SimdChunk::Width32(x $op y),
                    (SimdChunk::Width64(x), SimdChunk::Width64(y)) => SimdChunk::Width64(x $op y),
                    _ => panic!("Chunks must be of equal width"),
                }
            }
        }

        impl $trait for SimdChunk<$simd_type> {
            type Output = SimdChunk<$simd_type>;

            fn $method(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (SimdChunk::Width1(x), SimdChunk::Width1(y)) => SimdChunk::Width1(x $op y),
                    (SimdChunk::Width2(x), SimdChunk::Width2(y)) => SimdChunk::Width2(x $op y),
                    (SimdChunk::Width4(x), SimdChunk::Width4(y)) => SimdChunk::Width4(x $op y),
                    (SimdChunk::Width8(x), SimdChunk::Width8(y)) => SimdChunk::Width8(x $op y),
                    (SimdChunk::Width16(x), SimdChunk::Width16(y)) => SimdChunk::Width16(x $op y),
                    (SimdChunk::Width32(x), SimdChunk::Width32(y)) => SimdChunk::Width32(x $op y),
                    (SimdChunk::Width64(x), SimdChunk::Width64(y)) => SimdChunk::Width64(x $op y),
                    _ => panic!("Chunks must be of equal width"),
                }
            }
        }
    };
}

macro_rules! impl_simdchunk_op {
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
        impl Mul<$simd_type> for &SimdChunk<$simd_type> {
            type Output = SimdChunk<$simd_type>;

            fn mul(self, rhs: $simd_type) -> Self::Output {
                match self {
                    SimdChunk::Width1(x) => {
                        SimdChunk::Width1(x * Simd::<$simd_type, 1>::splat(rhs))
                    }
                    SimdChunk::Width2(x) => {
                        SimdChunk::Width2(x * Simd::<$simd_type, 2>::splat(rhs))
                    }
                    SimdChunk::Width4(x) => {
                        SimdChunk::Width4(x * Simd::<$simd_type, 4>::splat(rhs))
                    }
                    SimdChunk::Width8(x) => {
                        SimdChunk::Width8(x * Simd::<$simd_type, 8>::splat(rhs))
                    }
                    SimdChunk::Width16(x) => {
                        SimdChunk::Width16(x * Simd::<$simd_type, 16>::splat(rhs))
                    }
                    SimdChunk::Width32(x) => {
                        SimdChunk::Width32(x * Simd::<$simd_type, 32>::splat(rhs))
                    }
                    SimdChunk::Width64(x) => {
                        SimdChunk::Width64(x * Simd::<$simd_type, 64>::splat(rhs))
                    }
                }
            }
        }

        impl Mul<$simd_type> for SimdChunk<$simd_type> {
            type Output = SimdChunk<$simd_type>;

            fn mul(self, rhs: $simd_type) -> Self::Output {
                match self {
                    SimdChunk::Width1(x) => {
                        SimdChunk::Width1(x * Simd::<$simd_type, 1>::splat(rhs))
                    }
                    SimdChunk::Width2(x) => {
                        SimdChunk::Width2(x * Simd::<$simd_type, 2>::splat(rhs))
                    }
                    SimdChunk::Width4(x) => {
                        SimdChunk::Width4(x * Simd::<$simd_type, 4>::splat(rhs))
                    }
                    SimdChunk::Width8(x) => {
                        SimdChunk::Width8(x * Simd::<$simd_type, 8>::splat(rhs))
                    }
                    SimdChunk::Width16(x) => {
                        SimdChunk::Width16(x * Simd::<$simd_type, 16>::splat(rhs))
                    }
                    SimdChunk::Width32(x) => {
                        SimdChunk::Width32(x * Simd::<$simd_type, 32>::splat(rhs))
                    }
                    SimdChunk::Width64(x) => {
                        SimdChunk::Width64(x * Simd::<$simd_type, 64>::splat(rhs))
                    }
                }
            }
        }
    };
}

impl_simdchunk_op!(Add, +, add);
impl_simdchunk_op!(Sub, -, sub);
impl_simdchunk_op!(Mul, *, mul);

// Scalar products
impl_scalar_product_for_type!(f32);
impl_scalar_product_for_type!(f64);
impl_scalar_product_for_type!(i8);
impl_scalar_product_for_type!(i16);
impl_scalar_product_for_type!(i32);
impl_scalar_product_for_type!(i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_can_multiply() {
        let x = vec![1, 2, 3, 4];
        let y = vec![2, 3, 2, 3];

        let z = SimdChunk::from(&x[..]) * SimdChunk::from(&y[..]);
        assert_eq!(
            z,
            SimdChunk::Width4(Simd::<i32, 4>::from_array([2, 6, 6, 12]))
        );
    }

    #[test]
    fn it_can_add() {
        let x = vec![1, 2, 3, 4];
        let y = vec![2, 3, 2, 3];

        let z = SimdChunk::from(&x[..]) + SimdChunk::from(&y[..]);
        assert_eq!(
            z,
            SimdChunk::Width4(Simd::<i32, 4>::from_array([3, 5, 5, 7]))
        );
    }

    #[test]
    fn it_can_subtract() {
        let x = vec![1, 2, 3, 4];
        let y = vec![2, 3, 2, 3];

        let z = SimdChunk::from(&x[..]) - SimdChunk::from(&y[..]);
        assert_eq!(
            z,
            SimdChunk::Width4(Simd::<i32, 4>::from_array([-1, -1, 1, 1]))
        );
    }

    #[test]
    fn it_can_scalar_multiply() {
        let x = vec![1, 2, 3, 4];

        let z = SimdChunk::from(&x[..]) * 5;
        assert_eq!(
            z,
            SimdChunk::Width4(Simd::<i32, 4>::from_array([5, 10, 15, 20]))
        );
    }
}
