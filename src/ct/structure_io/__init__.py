"""
Structure and Omics I/O Module for CellType-Agent Phase 2.

Handles input/output for:
- PDB structure files
- h5ad single-cell data
- FASTA sequences
- Multi-modal input processing
"""

from ct.structure_io.pdb_handler import PDBHandler
from ct.structure_io.h5ad_handler import H5ADHandler
from ct.structure_io.fasta_handler import FASTAHandler

__all__ = [
    "PDBHandler",
    "H5ADHandler",
    "FASTAHandler",
]