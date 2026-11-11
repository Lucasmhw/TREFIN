from pathlib import Path

import yaml


def test_every_implemented_manual_requirement_has_live_code_and_tests() -> None:
    repository = Path(__file__).resolve().parents[1]
    contract = yaml.safe_load(
        (repository / "docs" / "manual_contract.yaml").read_text(encoding="utf-8")
    )
    allowed = set(contract["status_vocabulary"])
    identifiers = []
    for requirement in contract["requirements"]:
        identifiers.append(requirement["id"])
        assert requirement["status"] in allowed
        if requirement["status"] == "implemented":
            assert requirement["code"], requirement["id"]
            assert requirement["tests"], requirement["id"]
            for relative_path in requirement["code"] + requirement["tests"]:
                assert (repository / relative_path).is_file(), (
                    requirement["id"],
                    relative_path,
                )
    assert len(identifiers) == len(set(identifiers))


def test_contract_forbids_a_publishable_replication_claim_while_items_are_open() -> None:
    repository = Path(__file__).resolve().parents[1]
    contract = yaml.safe_load(
        (repository / "docs" / "manual_contract.yaml").read_text(encoding="utf-8")
    )
    status_by_id = {
        requirement["id"]: requirement["status"]
        for requirement in contract["requirements"]
    }
    required = contract["claim_rule"]["publishable_manual_replication_requires"]
    assert any(status_by_id[identifier] != "implemented" for identifier in required)
