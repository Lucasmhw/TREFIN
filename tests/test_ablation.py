import pytest

from fconcord.ablation import RETRAINED_ABLATIONS, run_retrained_ablation


def test_retrained_ablation_registry_contains_five_structural_controls() -> None:
    assert RETRAINED_ABLATIONS == (
        "NoGraph",
        "NoMarketState",
        "NoRiskConcepts",
        "NoForwardConceptSupervision",
        "GaussianSymmetricDecoder",
    )


def test_unknown_ablation_is_rejected_before_filesystem_access(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown retrained ablation"):
        run_retrained_ablation(tmp_path / "missing", "Invalid", tmp_path / "out")
