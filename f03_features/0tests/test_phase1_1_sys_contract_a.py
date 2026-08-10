# Run: python -m f03_features.0tests.test_phase1_1_sys_contract_a
"""
این تستر تقریباً تمام Contractهای موجود در فایل sys_contract.py را پوشش می‌دهد:
    - اعتبارسنجی موفق
    - خطای نسخه Engine
    - خطای نسخه Resolver
    - خطای نسخه Config
    - پایداری Signature
"""
from f03_features.feature_B_sys_contract import (
    SystemContract,
    ContractValidator,
    ContractGate,
    enforce_contract,
)


def test_valid_contract():

    contract = SystemContract(
        config_version="1.0.0",
        engine_version="1.0.0",
        resolver_version="1.0.0",
        mode="train",
    )

    validator = ContractValidator(
        engine_version="1.0.0",
        resolver_version="1.0.0",
        supported_config_versions={"1.0.0"},
    )

    gate = ContractGate(validator)

    enforce_contract(gate, contract)

    print("VALID CONTRACT TEST PASSED")


def test_engine_version_mismatch():

    contract = SystemContract(
        config_version="1.0.0",
        engine_version="9.9.9",
        resolver_version="1.0.0",
        mode="train",
    )

    validator = ContractValidator(
        engine_version="1.0.0",
        resolver_version="1.0.0",
        supported_config_versions={"1.0.0"},
    )

    gate = ContractGate(validator)

    try:
        enforce_contract(gate, contract)
        raise AssertionError("Expected ENGINE_VERSION_MISMATCH")

    except RuntimeError as e:
        assert str(e) == "ENGINE_VERSION_MISMATCH"

    print("ENGINE VERSION MISMATCH TEST PASSED")


def test_resolver_version_mismatch():

    contract = SystemContract(
        config_version="1.0.0",
        engine_version="1.0.0",
        resolver_version="9.9.9",
        mode="train",
    )

    validator = ContractValidator(
        engine_version="1.0.0",
        resolver_version="1.0.0",
        supported_config_versions={"1.0.0"},
    )

    gate = ContractGate(validator)

    try:
        enforce_contract(gate, contract)
        raise AssertionError("Expected RESOLVER_VERSION_MISMATCH")

    except RuntimeError as e:
        assert str(e) == "RESOLVER_VERSION_MISMATCH"

    print("RESOLVER VERSION MISMATCH TEST PASSED")


def test_config_version_not_supported():

    contract = SystemContract(
        config_version="999.0.0",
        engine_version="1.0.0",
        resolver_version="1.0.0",
        mode="train",
    )

    validator = ContractValidator(
        engine_version="1.0.0",
        resolver_version="1.0.0",
        supported_config_versions={"1.0.0"},
    )

    gate = ContractGate(validator)

    try:
        enforce_contract(gate, contract)
        raise AssertionError("Expected CONFIG_VERSION_NOT_SUPPORTED")

    except RuntimeError as e:
        assert str(e) == "CONFIG_VERSION_NOT_SUPPORTED"

    print("CONFIG VERSION TEST PASSED")


def test_signature_stability():

    c1 = SystemContract(
        config_version="1.0.0",
        engine_version="1.0.0",
        resolver_version="1.0.0",
        mode="train",
    )

    c2 = SystemContract(
        config_version="1.0.0",
        engine_version="1.0.0",
        resolver_version="1.0.0",
        mode="train",
    )

    assert c1.signature() == c2.signature()

    print("SIGNATURE STABILITY TEST PASSED")


if __name__ == "__main__":

    test_valid_contract()
    test_engine_version_mismatch()
    test_resolver_version_mismatch()
    test_config_version_not_supported()
    test_signature_stability()

    print("\nALL SYS_CONTRACT TESTS PASSED")