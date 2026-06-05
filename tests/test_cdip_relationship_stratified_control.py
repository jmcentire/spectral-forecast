"""Tests for the acquisition-stratified CDIP relationship control."""

from experiments.cdip_relationship_stratified_control import acquisition_stratum


def test_acquisition_stratum_combines_sample_rate_and_processing_family() -> None:
    assert acquisition_stratum(
        {"sample_rate": 1.279999971, "processing_family": "lx"}
    ) == "1.28:lx"
