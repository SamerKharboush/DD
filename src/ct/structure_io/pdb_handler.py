"""
PDB Handler for protein structure files.

Provides utilities for:
- Parsing PDB files
- Extracting sequences and coordinates
- Pocket detection
- Structure validation
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("ct.structure_io.pdb")


@dataclass
class PDBStructure:
    """Parsed PDB structure."""
    pdb_id: str
    sequence: str
    chains: list[str]
    residues: list[dict]
    coordinates: np.ndarray
    resolution: Optional[float] = None
    method: str = "X-RAY"
    header: dict = field(default_factory=dict)
    pockets: list[dict] = field(default_factory=list)


class PDBHandler:
    """
    Handler for PDB structure files.

    Usage:
        handler = PDBHandler()
        structure = handler.parse("target.pdb")
        pockets = handler.detect_pockets(structure)
    """

    # Standard amino acid 3-letter to 1-letter mapping
    AA_MAP = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        # Non-standard
        "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XAA": "X",
    }

    def __init__(self):
        """Initialize PDB handler."""
        pass

    def parse(self, pdb_path: Path | str) -> PDBStructure:
        """
        Parse a PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            PDBStructure object
        """
        pdb_path = Path(pdb_path)
        content = pdb_path.read_text()

        lines = content.split("\n")

        # Extract header info
        header = self._parse_header(lines)
        pdb_id = header.get("id", pdb_path.stem)

        # Extract sequence from SEQRES or ATOM records
        sequence = self._extract_sequence(lines)

        # Extract chains
        chains = self._extract_chains(lines)

        # Extract residues with coordinates
        residues, coordinates = self._extract_residues(lines)

        # Resolution
        resolution = self._extract_resolution(lines)

        # Method
        method = self._extract_method(lines)

        return PDBStructure(
            pdb_id=pdb_id,
            sequence=sequence,
            chains=chains,
            residues=residues,
            coordinates=coordinates,
            resolution=resolution,
            method=method,
            header=header,
        )

    def _parse_header(self, lines: list[str]) -> dict:
        """Parse PDB header."""
        header = {}

        for line in lines:
            if line.startswith("HEADER"):
                header["classification"] = line[10:50].strip()
                header["date"] = line[50:59].strip()
                header["id"] = line[62:66].strip()
            elif line.startswith("TITLE"):
                header["title"] = line[10:].strip()
            elif line.startswith("COMPND"):
                header["compound"] = header.get("compound", "") + line[10:].strip()

        return header

    def _extract_sequence(self, lines: list[str]) -> str:
        """Extract protein sequence from PDB."""
        # Try SEQRES first
        seqres_seq = ""
        current_chain = None

        for line in lines:
            if line.startswith("SEQRES"):
                chain = line[11].strip()
                if current_chain is None:
                    current_chain = chain

                if chain == current_chain:
                    residues = line[19:].split()
                    for res in residues:
                        seqres_seq += self.AA_MAP.get(res, "X")

        if seqres_seq:
            return seqres_seq

        # Fall back to ATOM records
        seen_residues = set()
        sequence = ""

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_num = line[22:26].strip()

                key = (chain, res_num)
                if key not in seen_residues:
                    seen_residues.add(key)
                    sequence += self.AA_MAP.get(res_name, "X")

        return sequence

    def _extract_chains(self, lines: list[str]) -> list[str]:
        """Extract chain IDs."""
        chains = set()

        for line in lines:
            if line.startswith("ATOM") or line.startswith("SEQRES"):
                chain = line[21].strip() if line.startswith("ATOM") else line[11].strip()
                if chain:
                    chains.add(chain)

        return sorted(list(chains))

    def _extract_residues(self, lines: list[str]) -> tuple[list[dict], np.ndarray]:
        """Extract residue information and coordinates."""
        residues = []
        coords = []
        seen_residues = set()

        for line in lines:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                key = (chain, res_num)

                if key not in seen_residues:
                    seen_residues.add(key)
                    residues.append({
                        "chain": chain,
                        "number": res_num,
                        "name": res_name,
                        "aa": self.AA_MAP.get(res_name, "X"),
                    })

                coords.append([x, y, z])

        coordinates = np.array(coords) if coords else np.array([]).reshape(0, 3)

        return residues, coordinates

    def _extract_resolution(self, lines: list[str]) -> Optional[float]:
        """Extract resolution from REMARK records."""
        for line in lines:
            if line.startswith("REMARK   2 RESOLUTION"):
                try:
                    return float(line[23:].strip().replace("ANGSTROMS", "").strip())
                except ValueError:
                    pass
        return None

    def _extract_method(self, lines: list[str]) -> str:
        """Extract experimental method."""
        for line in lines:
            if line.startswith("EXPDTA"):
                method = line[6:].strip().upper()
                if "X-RAY" in method:
                    return "X-RAY"
                elif "NMR" in method:
                    return "NMR"
                elif "CRYO-EM" in method or "ELECTRON MICROSCOPY" in method:
                    return "CRYO-EM"
        return "UNKNOWN"

    def detect_pockets(
        self,
        structure: PDBStructure,
        min_volume: float = 100.0,
    ) -> list[dict]:
        """
        Detect potential binding pockets.

        Args:
            structure: PDB structure
            min_volume: Minimum pocket volume in A^3

        Returns:
            List of detected pockets
        """
        # Simple pocket detection based on residue clustering
        # In practice, would use fpocket or similar tool

        pockets = []

        if structure.residues:
            # Divide structure into segments
            n_residues = len(structure.residues)
            segment_size = max(10, n_residues // 5)

            for i in range(0, n_residues, segment_size):
                end = min(i + segment_size, n_residues)
                segment_residues = structure.residues[i:end]

                center = self._calculate_centroid(structure, i, end)

                pockets.append({
                    "id": f"pocket_{len(pockets) + 1}",
                    "center": center.tolist() if isinstance(center, np.ndarray) else center,
                    "residues": [r["number"] for r in segment_residues],
                    "volume_estimate": segment_size * 15,  # Rough estimate
                })

        return [p for p in pockets if p["volume_estimate"] >= min_volume]

    def _calculate_centroid(
        self,
        structure: PDBStructure,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """Calculate centroid of residue range."""
        # Simplified - would use actual coordinates
        if len(structure.coordinates) > 0:
            start_coord = min(start_idx * 5, len(structure.coordinates))
            end_coord = min(end_idx * 5, len(structure.coordinates))
            if start_coord < end_coord:
                return np.mean(structure.coordinates[start_coord:end_coord], axis=0)

        return np.array([0.0, 0.0, 0.0])

    def extract_sequence_for_chain(
        self,
        structure: PDBStructure,
        chain_id: str,
    ) -> str:
        """Extract sequence for a specific chain."""
        sequence = ""

        for res in structure.residues:
            if res["chain"] == chain_id:
                sequence += res["aa"]

        return sequence

    def get_interface_residues(
        self,
        structure: PDBStructure,
        chain1: str,
        chain2: str,
        distance_threshold: float = 5.0,
    ) -> list[tuple]:
        """
        Get interface residues between two chains.

        Args:
            structure: PDB structure
            chain1: First chain ID
            chain2: Second chain ID
            distance_threshold: Distance threshold in Angstroms

        Returns:
            List of (residue1, residue2) pairs at interface
        """
        # Simplified interface detection
        # Would use actual coordinates for proper calculation
        interface = []

        res1 = [r for r in structure.residues if r["chain"] == chain1]
        res2 = [r for r in structure.residues if r["chain"] == chain2]

        # Simplified: assume residues near chain boundaries are interface
        if res1 and res2:
            # Take last 10 of chain1 and first 10 of chain2
            for r1 in res1[-10:]:
                for r2 in res2[:10]:
                    interface.append((r1["number"], r2["number"]))

        return interface

    def validate_structure(self, structure: PDBStructure) -> dict:
        """
        Validate PDB structure.

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Check sequence length
        if len(structure.sequence) < 10:
            issues.append("Sequence too short")
        elif len(structure.sequence) > 2000:
            warnings.append("Sequence unusually long")

        # Check resolution
        if structure.resolution:
            if structure.resolution > 3.0:
                warnings.append(f"Low resolution: {structure.resolution:.1f}Å")
            elif structure.resolution > 2.5:
                warnings.append(f"Medium resolution: {structure.resolution:.1f}Å")

        # Check for missing residues (gaps in numbering)
        numbers = [r["number"] for r in structure.residues]
        if numbers:
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] > 1:
                    warnings.append(f"Gap detected at residue {numbers[i]}")

        # Check for unknown residues
        unknown_count = sum(1 for r in structure.residues if r["aa"] == "X")
        if unknown_count > 0:
            warnings.append(f"{unknown_count} unknown/modified residues")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": {
                "sequence_length": len(structure.sequence),
                "num_chains": len(structure.chains),
                "num_residues": len(structure.residues),
                "resolution": structure.resolution,
                "method": structure.method,
            },
        }

    def to_fasta(self, structure: PDBStructure) -> str:
        """Convert PDB structure to FASTA format."""
        return f">{structure.pdb_id}\n{structure.sequence}\n"


def parse_pdb_file(pdb_path: str, **kwargs) -> dict:
    """
    Parse a PDB structure file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary with parsed structure
    """
    handler = PDBHandler()
    structure = handler.parse(pdb_path)
    validation = handler.validate_structure(structure)

    return {
        "summary": (
            f"Parsed PDB {structure.pdb_id}: {len(structure.sequence)} residues, "
            f"{len(structure.chains)} chains, resolution {structure.resolution or 'N/A'}Å"
        ),
        "pdb_id": structure.pdb_id,
        "sequence": structure.sequence,
        "sequence_length": len(structure.sequence),
        "chains": structure.chains,
        "resolution": structure.resolution,
        "method": structure.method,
        "validation": validation,
        "pockets": handler.detect_pockets(structure),
    }