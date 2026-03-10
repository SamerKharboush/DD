"""
Structure I/O Tools for CellType-Agent Phase 2.

Registers tools for handling various file inputs:
- PDB structure files
- h5ad single-cell data
- FASTA sequences
"""

from ct.tools import registry


# ============================================================
# STRUCTURE I/O TOOLS
# ============================================================

@registry.register(
    name="structure.parse_pdb",
    description="Parse a PDB structure file and extract sequence, chains, and pocket information",
    category="structure",
    parameters={
        "pdb_path": "Path to PDB file",
    },
)
def structure_parse_pdb(pdb_path: str, **kwargs) -> dict:
    """Parse a PDB structure file."""
    from ct.structure_io.pdb_handler import parse_pdb_file

    return parse_pdb_file(pdb_path)


@registry.register(
    name="structure.detect_pockets",
    description="Detect potential binding pockets in a protein structure",
    category="structure",
    parameters={
        "pdb_path": "Path to PDB file",
        "min_volume": "Minimum pocket volume in cubic Angstroms (default: 100)",
    },
)
def structure_detect_pockets(pdb_path: str, min_volume: float = 100.0, **kwargs) -> dict:
    """Detect binding pockets in a structure."""
    from ct.structure_io.pdb_handler import PDBHandler

    handler = PDBHandler()
    structure = handler.parse(pdb_path)
    pockets = handler.detect_pockets(structure, min_volume)

    return {
        "summary": f"Detected {len(pockets)} potential binding pockets",
        "pockets": [
            {
                "id": p["id"],
                "center": p["center"],
                "residues": p["residues"][:10],  # First 10 residues
                "volume_estimate": p["volume_estimate"],
            }
            for p in pockets
        ],
    }


@registry.register(
    name="structure.analyze_h5ad",
    description="Analyze an h5ad single-cell RNA-seq data file",
    category="omics",
    parameters={
        "h5ad_path": "Path to h5ad file",
    },
)
def structure_analyze_h5ad(h5ad_path: str, **kwargs) -> dict:
    """Analyze h5ad single-cell data."""
    from ct.structure_io.h5ad_handler import analyze_h5ad_file

    return analyze_h5ad_file(h5ad_path)


@registry.register(
    name="structure.extract_expression",
    description="Extract expression values for specific genes from h5ad file",
    category="omics",
    parameters={
        "h5ad_path": "Path to h5ad file",
        "genes": "Comma-separated list of gene symbols",
    },
)
def structure_extract_expression(h5ad_path: str, genes: str, **kwargs) -> dict:
    """Extract gene expression from h5ad file."""
    from ct.structure_io.h5ad_handler import extract_gene_expression

    return extract_gene_expression(h5ad_path, genes)


@registry.register(
    name="structure.parse_fasta",
    description="Parse a FASTA sequence file",
    category="structure",
    parameters={
        "fasta_path": "Path to FASTA file",
    },
)
def structure_parse_fasta(fasta_path: str, **kwargs) -> dict:
    """Parse a FASTA file."""
    from ct.structure_io.fasta_handler import parse_fasta_file

    return parse_fasta_file(fasta_path)


@registry.register(
    name="structure.translate_dna",
    description="Translate DNA sequence to protein",
    category="structure",
    parameters={
        "dna_sequence": "DNA sequence string",
    },
)
def structure_translate_dna(dna_sequence: str, **kwargs) -> dict:
    """Translate DNA to protein."""
    from ct.structure_io.fasta_handler import FASTAHandler

    handler = FASTAHandler()
    protein = handler.translate_dna(dna_sequence)

    return {
        "summary": f"Translated {len(dna_sequence)} bp DNA to {len(protein)} aa protein",
        "protein_sequence": protein,
        "dna_length": len(dna_sequence),
        "protein_length": len(protein),
    }


# Register all tools
logger = __import__("logging").getLogger("ct.tools.structure_io")
logger.info("Structure I/O tools registered: pdb, h5ad, fasta")