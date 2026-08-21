from pymatgen.core import Element
from pymatgen.entries.compatibility import Compatibility, CompatibilityError
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,
    EnergyAdjustment,
)


class IdentityCorrectionScheme(Compatibility):
    """Perform no energy correction."""

    def get_adjustments(
        self, entry: ComputedEntry | ComputedStructureEntry
    ) -> list[EnergyAdjustment]:
        return []


class TRI110Compatibility2024(Compatibility):
    """This is an implementation of the correction scheme defined in

    A Simple Linear Relation Solves Unphysical DFT Energy Corrections
    B. A. Rohr, S. K. Suram, J. S. Bakander, ChemRxiv, 10.26434/chemrxiv-2024-q5058, (2024)

    https://chemrxiv.org/engage/chemrxiv/article-details/67252d617be152b1d0b2c1ef
    """

    # Compatibility.name needed for compatibility with CorrectedEntriesBuilder.process_item.
    name: str = "TRI110Compatibility2024"

    # See Section 2.1 of
    # https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/672533a35a82cea2fac0b474/original/supplemental-information-a-simple-linear-relation-solves-unphysical-dft-energy-corrections.pdf
    PBE_CORRECTION: float = 1.108

    # See Table 1 of
    # https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/672533a35a82cea2fac0b474/original/supplemental-information-a-simple-linear-relation-solves-unphysical-dft-energy-corrections.pdf
    U_CORRECTION = {
        Element("Co"): -2.275,
        Element("Cr"): -2.707,
        Element("Fe"): -3.189,
        Element("Mn"): -2.28,
        Element("Mo"): -4.93,
        Element("Ni"): -3.361,
        Element("V"): -2.774,
        Element("W"): -6.261,
    }

    def get_adjustments(
        self, entry: ComputedEntry | ComputedStructureEntry
    ) -> list[EnergyAdjustment]:
        """Get the energy adjustments for a ComputedEntry.

        This method must generate a list of EnergyAdjustment objects
        of the appropriate type (constant, composition-based, or temperature-based)
        to be applied to the ComputedEntry, and must raise a CompatibilityError
        if the entry is not compatible.

        Args:
            entry: A ComputedEntry object.

        Returns:
            list[EnergyAdjustment]: A list of EnergyAdjustment to be applied to the
                Entry, which are evaluated in ComputedEntry.correction. Note that
                the later implements a linear sum of corrections.

        Raises:
            CompatibilityError if the entry is not compatible

        """
        if entry.parameters.get("run_type") not in ("GGA", "GGA+U"):
            raise CompatibilityError(
                f"Entry {entry.entry_id} has invalid run type {entry.parameters.get('run_type')}. "
                f"Must be GGA or GGA+U. Discarding."
            )

        adjustments = []

        if entry.parameters.get("run_type") in ["GGA", "GGA+U"]:
            # multiplicative adjust for all PBE or PBE+U calculations
            # energy adjustments are applied additively in downstram pymatgen code, so
            # refactor multiplicate factor as an addition to uncorrected energy
            adjustments.append(
                EnergyAdjustment(
                    value=entry.uncorrected_energy * (self.PBE_CORRECTION - 1.0),
                    name="TRI110PBE",
                )
            )

        if entry.parameters.get("run_type") == "GGA+U":
            u_elements = [el for el in entry.composition if el in self.U_CORRECTION]

            # number of atoms of each element
            composition_dict: dict[str, float] = entry.composition.as_dict()

            # eV
            # EnergyAdjustment(value) expects the total energy, so we multiply the
            # correction per U atom by the number of atoms of that type and not the
            # fractional composition.
            u_correction = sum(
                [composition_dict[el.name] * self.U_CORRECTION[el] for el in u_elements]
            )

            # EnergyAdjustment(value) assumes total energy
            adjustments.append(EnergyAdjustment(value=u_correction, name="TRI110PBE_U"))

        return adjustments