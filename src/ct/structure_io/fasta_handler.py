"""
FASTA Handler for protein and nucleotide sequences.

Provides utilities for:
- Parsing FASTA files
- Sequence validation
- Format conversion
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.structure_io.fasta")


@dataclass
class FASTASequence:
    """A parsed FASTA sequence."""
    header: str
    sequence: str
    sequence_type: str  # protein, DNA, RNA
    length: int
    description: Optional[str] = None


class FASTAHandler:
    """
    Handler for FASTA sequence files.

    Usage:
        handler = FASTAHandler()
        sequences = handler.parse("proteins.fasta")
        handler.write(sequences, "output.fasta")
    """

    def __init__(self):
        """Initialize FASTA handler."""
        pass

    def parse(self, fasta_path: Path | str) -> list[FASTASequence]:
        """
        Parse a FASTA file.

        Args:
            fasta_path: Path to FASTA file

        Returns:
            List of FASTASequence objects
        """
        fasta_path = Path(fasta_path)
        content = fasta_path.read_text()

        sequences = []
        current_header = None
        current_sequence = []

        for line in content.split("\n"):
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence
                if current_header is not None:
                    seq_str = "".join(current_sequence)
                    sequences.append(FASTASequence(
                        header=current_header,
                        sequence=seq_str,
                        sequence_type=self._detect_sequence_type(seq_str),
                        length=len(seq_str),
                        description=self._extract_description(current_header),
                    ))

                # Start new sequence
                current_header = line[1:]  # Remove >
                current_sequence = []
            else:
                current_sequence.append(line)

        # Save last sequence
        if current_header is not None:
            seq_str = "".join(current_sequence)
            sequences.append(FASTASequence(
                header=current_header,
                sequence=seq_str,
                sequence_type=self._detect_sequence_type(seq_str),
                length=len(seq_str),
                description=self._extract_description(current_header),
            ))

        return sequences

    def _detect_sequence_type(self, sequence: str) -> str:
        """Detect if sequence is protein, DNA, or RNA."""
        if not sequence:
            return "unknown"

        # Count valid characters for each type
        protein_chars = set("ACDEFGHIKLMNPQRSTVWYXUBZ")
        dna_chars = set("ACGTN")
        rna_chars = set("ACGUN")

        upper_seq = sequence.upper()
        total = len(upper_seq)

        protein_count = sum(1 for c in upper_seq if c in protein_chars)
        dna_count = sum(1 for c in upper_seq if c in dna_chars)
        rna_count = sum(1 for c in upper_seq if c in rna_chars)

        # Check for amino acid-specific characters
        has_aa_specific = any(c in "DEFHIKLMPQRSVWY" for c in upper_seq)

        if has_aa_specific:
            return "protein"
        elif "U" in upper_seq:
            return "RNA"
        elif dna_count / total > 0.9:
            return "DNA"
        else:
            return "protein"  # Default to protein

    def _extract_description(self, header: str) -> Optional[str]:
        """Extract description from header."""
        # Common formats: "ID description" or just "description"
        parts = header.split(None, 1)
        return parts[1] if len(parts) > 1 else None

    def validate(self, sequence: FASTASequence) -> dict:
        """
        Validate a FASTA sequence.

        Args:
            sequence: FASTASequence to validate

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Check sequence length
        if sequence.length == 0:
            issues.append("Empty sequence")
        elif sequence.length < 10:
            warnings.append("Very short sequence")
        elif sequence.length > 100000:
            warnings.append("Very long sequence")

        # Check for valid characters
        valid_chars = {
            "protein": set("ACDEFGHIKLMNPQRSTVWYXUBZ"),
            "DNA": set("ACGTN"),
            "RNA": set("ACGUN"),
            "unknown": set("ACDEFGHIKLMNPQRSTVWYXUBZACGTNU"),
        }

        invalid = set(sequence.sequence.upper()) - valid_chars.get(
            sequence.sequence_type, valid_chars["unknown"]
        )

        if invalid:
            issues.append(f"Invalid characters: {invalid}")

        # Check for stop codons in protein sequences
        if sequence.sequence_type == "protein":
            stop_count = sequence.sequence.count("*")
            if stop_count > 1:
                warnings.append(f"Multiple stop codons ({stop_count})")
            elif stop_count == 1 and not sequence.sequence.endswith("*"):
                warnings.append("Internal stop codon")

        # Check for unusual patterns
        if any(c * 10 in sequence.sequence for c in "ACDEFGHIKLMNPQRSTVWY"):
            warnings.append("Unusual repeat pattern detected")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "sequence_type": sequence.sequence_type,
            "length": sequence.length,
        }

    def write(
        self,
        sequences: list[FASTASequence],
        output_path: Path | str,
        line_width: int = 60,
    ) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            sequences: List of sequences to write
            output_path: Output file path
            line_width: Characters per line
        """
        output_path = Path(output_path)
        lines = []

        for seq in sequences:
            lines.append(f">{seq.header}")
            # Wrap sequence
            for i in range(0, len(seq.sequence), line_width):
                lines.append(seq.sequence[i:i + line_width])

        output_path.write_text("\n".join(lines) + "\n")

    def translate_dna(self, dna_sequence: str) -> str:
        """
        Translate DNA sequence to protein.

        Args:
            dna_sequence: DNA sequence string

        Returns:
            Protein sequence
        """
        codon_table = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }

        protein = []
        dna = dna_sequence.upper()

        for i in range(0, len(dna) - 2, 3):
            codon = dna[i:i + 3]
            aa = codon_table.get(codon, "X")
            if aa == "*":
                break
            protein.append(aa)

        return "".join(protein)

    def reverse_complement(self, dna_sequence: str) -> str:
        """
        Get reverse complement of DNA sequence.

        Args:
            dna_sequence: DNA sequence

        Returns:
            Reverse complement
        """
        complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        return "".join(complement.get(c, "N") for c in dna_sequence.upper()[::-1])

    def extract_region(
        self,
        sequence: FASTASequence,
        start: int,
        end: int,
    ) -> FASTASequence:
        """
        Extract a region from a sequence.

        Args:
            sequence: Source sequence
            start: Start position (1-indexed)
            end: End position (inclusive, 1-indexed)

        Returns:
            New FASTASequence with extracted region
        """
        # Convert to 0-indexed
        extracted = sequence.sequence[start - 1:end]

        return FASTASequence(
            header=f"{sequence.header}_{start}-{end}",
            sequence=extracted,
            sequence_type=sequence.sequence_type,
            length=len(extracted),
            description=f"Region {start}-{end} of {sequence.header}",
        )


def parse_fasta_file(fasta_path: str, **kwargs) -> dict:
    """
    Parse a FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dictionary with parsed sequences
    """
    handler = FASTAHandler()
    sequences = handler.parse(fasta_path)

    if not sequences:
        return {
            "summary": "No sequences found in FASTA file",
            "sequences": [],
        }

    # Validate all sequences
    validations = [handler.validate(seq) for seq in sequences]

    # Summary stats
    total_length = sum(seq.length for seq in sequences)
    avg_length = total_length / len(sequences)
    types = set(seq.sequence_type for seq in sequences)

    return {
        "summary": (
            f"Parsed {len(sequences)} sequences from FASTA. "
            f"Average length: {avg_length:.0f}. "
            f"Types: {', '.join(types)}"
        ),
        "n_sequences": len(sequences),
        "total_length": total_length,
        "average_length": avg_length,
        "sequence_types": list(types),
        "sequences": [
            {
                "header": seq.header,
                "length": seq.length,
                "type": seq.sequence_type,
                "valid": v["valid"],
                "issues": v["issues"],
            }
            for seq, v in zip(sequences, validations)
        ],
    }