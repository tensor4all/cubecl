use cubecl_ir::{ConstantValue, ElemType, FloatKind, VariableKind};
#[cfg(not(feature = "fp4"))]
use cubecl_ir::{StorageType, Type};

#[test]
fn float_metadata_ranges_remain_available_without_conversion_features() {
    for (kind, min, max) in [
        (FloatKind::E2M1, -6.0, 6.0),
        (FloatKind::E4M3, -416.0, 416.0),
        (FloatKind::E5M2, -49152.0, 49152.0),
        (FloatKind::UE8M0, 0.0, f64::from_bits(0x47e0000000000000)),
    ] {
        let elem = ElemType::Float(kind);
        assert_eq!(
            elem.min_variable().kind,
            VariableKind::Constant(ConstantValue::Float(min))
        );
        assert_eq!(
            elem.max_variable().kind,
            VariableKind::Constant(ConstantValue::Float(max))
        );
        assert_eq!(elem.epsilon(), 0.5 * (max - min));
    }
}

#[cfg(feature = "fp4")]
#[test]
fn fp4_range_and_cast_match_common_type() {
    use cubecl_common::e2m1;
    assert_eq!((e2m1::MIN, e2m1::MAX), (-6.0, 6.0));
    assert_eq!(
        ConstantValue::Float(1.3).cast_to(FloatKind::E2M1),
        ConstantValue::Float(e2m1::from_f64(1.3).to_f64())
    );
}

#[cfg(not(feature = "fp4"))]
#[test]
fn fp4_casts_require_feature_including_packed() {
    let value = ConstantValue::Float(1.0);
    assert!(std::panic::catch_unwind(|| value.cast_to(FloatKind::E2M1)).is_err());
    assert!(
        std::panic::catch_unwind(|| value.cast_to(Type::new(StorageType::Packed(
            ElemType::Float(FloatKind::E2M1),
            2
        ))))
        .is_err()
    );
}

#[cfg(feature = "fp8")]
#[test]
fn fp8_ranges_and_casts_match_common_types() {
    use cubecl_common::{e4m3, e5m2, ue8m0};
    assert_eq!((e4m3::MIN, e4m3::MAX), (-416.0, 416.0));
    assert_eq!((e5m2::MIN, e5m2::MAX), (-49152.0, 49152.0));
    assert_eq!(
        (ue8m0::MIN, ue8m0::MAX),
        (0.0, f64::from_bits(0x47e0000000000000))
    );
    for (kind, expected) in [
        (FloatKind::E4M3, e4m3::from_f64(1.3).to_f64()),
        (FloatKind::E5M2, e5m2::from_f64(1.3).to_f64()),
        (FloatKind::UE8M0, ue8m0::from_f64(1.3).to_f64()),
    ] {
        assert_eq!(
            ConstantValue::Float(1.3).cast_to(kind),
            ConstantValue::Float(expected)
        );
    }
}

#[cfg(not(feature = "fp8"))]
#[test]
fn fp8_casts_require_feature() {
    for kind in [FloatKind::E4M3, FloatKind::E5M2, FloatKind::UE8M0] {
        assert!(std::panic::catch_unwind(|| ConstantValue::Float(1.0).cast_to(kind)).is_err());
    }
}
