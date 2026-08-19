# Run: python -m pytest f03_features/0tests/test_phase1_1_sys_contract_b

import pytest

from f03_features.feature_B_sys_contract import (
    SystemContract,
    ContractValidator,
    ContractGate,
    enforce_contract,
)


def build_contract():
    return SystemContract(
        config_version="1.0",
        engine_version="3.0",
        resolver_version="2.0",
        mode="train",
    )


def build_gate():
    validator = ContractValidator(
        engine_version="3.0",
        resolver_version="2.0",
        supported_config_versions={"1.0"},
    )
    return ContractGate(validator)


def test_signature_is_deterministic():
    c1 = build_contract()
    c2 = build_contract()

    assert c1.signature() == c2.signature()


def test_valid_contract_passes():
    gate = build_gate()
    enforce_contract(gate, build_contract())


def test_engine_version_mismatch():
    gate = build_gate()

    bad = SystemContract(
        config_version="1.0",
        engine_version="999",
        resolver_version="2.0",
        mode="train",
    )

    with pytest.raises(RuntimeError, match="ENGINE_VERSION_MISMATCH"):
        enforce_contract(gate, bad)


def test_resolver_version_mismatch():
    gate = build_gate()

    bad = SystemContract(
        config_version="1.0",
        engine_version="3.0",
        resolver_version="999",
        mode="train",
    )

    with pytest.raises(RuntimeError, match="RESOLVER_VERSION_MISMATCH"):
        enforce_contract(gate, bad)


def test_config_version_not_supported():
    gate = build_gate()

    bad = SystemContract(
        config_version="999",
        engine_version="3.0",
        resolver_version="2.0",
        mode="train",
    )

    with pytest.raises(RuntimeError, match="CONFIG_VERSION_NOT_SUPPORTED"):
        enforce_contract(gate, bad)