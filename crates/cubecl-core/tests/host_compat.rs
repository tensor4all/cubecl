#![cfg(not(target_os = "none"))]

use cubecl_core::{
    CubeElement, e2m1, e2m1x2, e4m3, e5m2,
    frontend::{CubePrimitive, DynamicScalar},
    num_traits::{Float, ToPrimitive},
    ue8m0,
};

#[test]
fn minifloat_types_remain_available_without_core_defaults() {
    fn primitive<T: CubePrimitive>() {}
    fn element<T: CubeElement>() {}
    primitive::<e2m1>();
    primitive::<e2m1x2>();
    primitive::<e4m3>();
    primitive::<e5m2>();
    primitive::<ue8m0>();
    element::<e2m1x2>();
    element::<e4m3>();
    element::<e5m2>();
    element::<ue8m0>();
}

#[test]
fn dynamic_scalar_uses_host_math() {
    let x = 0.031851851562_f64;
    let scalar = DynamicScalar::<()>::from_f64(x);
    assert_eq!(
        Float::sin(scalar).to_f64().unwrap().to_bits(),
        x.sin().to_bits()
    );
}
