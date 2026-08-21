import pytest
from pymatgen.core import Element
from pymatgen.entries.computed_entries import ComputedEntry, EnergyAdjustment

from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024


def test_tri_pbe_correction_uses_uncorrected_energy() -> None:
    entry = ComputedEntry(
        composition="NaCl",
        energy=-6.77796,
        parameters={"run_type": "GGA"},
    )
    entry.energy_adjustments.append(
        EnergyAdjustment(value=-0.614, name="MP2020 anion correction (Cl)")
    )

    adjustments = TRI110Compatibility2024().get_adjustments(entry)

    assert len(adjustments) == 1
    assert adjustments[0].name == "TRI110PBE"
    assert adjustments[0].value == pytest.approx(0.108 * entry.uncorrected_energy)


def test_tri_u_correction_is_separate_from_pbe_scaling() -> None:
    entry = ComputedEntry(
        composition="FeO",
        energy=-10.0,
        parameters={"run_type": "GGA+U"},
    )

    adjustments = TRI110Compatibility2024().get_adjustments(entry)

    assert {adjustment.name: adjustment.value for adjustment in adjustments} == pytest.approx(
        {
            "TRI110PBE": -1.08,
            "TRI110PBE_U": TRI110Compatibility2024.U_CORRECTION[Element("Fe")],
        }
    )
