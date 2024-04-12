# SIMD Vec

Arbitrary sized SIMD vectors for Rust.

**WARNING:** Only works in Rust Nightly.

## Example

```rs
use simd_vec::SimdVec;

fn main() {
    let x = vec![1, 2, 3, 4, 5, 6];
    let y = vec![2, 2, 2, 2, 2, 2];
    let z: SimdVec<i32> = SimdVec::from(x) * SimdVec::from(y);
    println!("{:?}", z.into_vec());
}
```
