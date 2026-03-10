"""
h5ad Handler for single-cell RNA-seq data.

Provides utilities for:
- Loading AnnData objects
- Extracting expression matrices
- Cell type annotation
- Perturbation prediction preparation
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ct.structure_io.h5ad")


@dataclass
class H5ADSummary:
    """Summary of h5ad file."""
    file_path: str
    n_cells: int
    n_genes: int
    cell_types: list[str]
    obs_columns: list[str]
    var_columns: list[str]
    has_raw: bool
    has_spatial: bool
    cell_type_counts: dict = field(default_factory=dict)


class H5ADHandler:
    """
    Handler for h5ad (AnnData) single-cell data files.

    Usage:
        handler = H5ADHandler()
        summary = handler.summarize("sample.h5ad")
        matrix = handler.extract_expression("sample.h5ad", genes=["TP53", "KRAS"])
    """

    def __init__(self):
        """Initialize h5ad handler."""
        pass

    def load(self, h5ad_path: Path | str):
        """
        Load an h5ad file.

        Args:
            h5ad_path: Path to h5ad file

        Returns:
            AnnData object
        """
        try:
            import scanpy as sc
            return sc.read_h5ad(h5ad_path)
        except ImportError:
            raise ImportError(
                "scanpy required for h5ad support. "
                "Install with: pip install scanpy anndata"
            )

    def summarize(self, h5ad_path: Path | str) -> H5ADSummary:
        """
        Get summary of h5ad file.

        Args:
            h5ad_path: Path to h5ad file

        Returns:
            H5ADSummary object
        """
        adata = self.load(h5ad_path)

        # Get cell types if available
        cell_types = []
        cell_type_counts = {}

        if "cell_type" in adata.obs.columns:
            cell_types = adata.obs["cell_type"].unique().tolist()
            cell_type_counts = adata.obs["cell_type"].value_counts().to_dict()
        elif "celltype" in adata.obs.columns:
            cell_types = adata.obs["celltype"].unique().tolist()
            cell_type_counts = adata.obs["celltype"].value_counts().to_dict()

        # Check for raw layer
        has_raw = adata.raw is not None

        # Check for spatial data
        has_spatial = "spatial" in adata.uns or (
            "spatial" in adata.obsm if adata.obsm is not None else False
        )

        return H5ADSummary(
            file_path=str(h5ad_path),
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            cell_types=cell_types,
            obs_columns=adata.obs.columns.tolist(),
            var_columns=adata.var.columns.tolist(),
            has_raw=has_raw,
            has_spatial=has_spatial,
            cell_type_counts=cell_type_counts,
        )

    def extract_expression(
        self,
        h5ad_path: Path | str,
        genes: list[str],
        layer: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract expression matrix for specific genes.

        Args:
            h5ad_path: Path to h5ad file
            genes: List of gene names
            layer: Layer to extract from (None for .X)

        Returns:
            DataFrame with cells x genes
        """
        adata = self.load(h5ad_path)

        # Find gene indices
        gene_mask = adata.var_names.isin(genes)

        if not gene_mask.any():
            logger.warning(f"None of the requested genes found in data")
            return pd.DataFrame()

        # Extract expression
        if layer and layer in adata.layers:
            expr_matrix = adata.layers[layer][:, gene_mask]
        else:
            expr_matrix = adata.X[:, gene_mask]

        # Convert to dense if sparse
        if hasattr(expr_matrix, "toarray"):
            expr_matrix = expr_matrix.toarray()

        # Create DataFrame
        found_genes = adata.var_names[gene_mask].tolist()

        return pd.DataFrame(
            expr_matrix,
            index=adata.obs_names,
            columns=found_genes,
        )

    def get_cell_metadata(
        self,
        h5ad_path: Path | str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get cell metadata.

        Args:
            h5ad_path: Path to h5ad file
            columns: Specific columns to extract (None for all)

        Returns:
            DataFrame with cell metadata
        """
        adata = self.load(h5ad_path)

        if columns:
            available = [c for c in columns if c in adata.obs.columns]
            return adata.obs[available]
        return adata.obs

    def filter_cells(
        self,
        h5ad_path: Path | str,
        filters: dict,
        output_path: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """
        Filter cells based on criteria.

        Args:
            h5ad_path: Path to h5ad file
            filters: Dictionary of column -> value filters
            output_path: Optional path to save filtered data

        Returns:
            Filtered AnnData or DataFrame
        """
        adata = self.load(h5ad_path)

        mask = pd.Series([True] * adata.n_obs, index=adata.obs_names)

        for col, values in filters.items():
            if col in adata.obs.columns:
                if isinstance(values, list):
                    mask &= adata.obs[col].isin(values)
                else:
                    mask &= adata.obs[col] == values

        filtered = adata[mask]

        if output_path:
            filtered.write_h5ad(output_path)
            logger.info(f"Saved filtered data to {output_path}")

        return filtered

    def prepare_for_perturbation(
        self,
        h5ad_path: Path | str,
        treatment_column: str = "treatment",
        control_value: str = "control",
    ) -> dict:
        """
        Prepare data for perturbation prediction.

        Args:
            h5ad_path: Path to h5ad file
            treatment_column: Column with treatment labels
            control_value: Value for control samples

        Returns:
            Dictionary with control and treatment data
        """
        adata = self.load(h5ad_path)

        # Split control and treated
        if treatment_column not in adata.obs.columns:
            raise ValueError(f"Treatment column '{treatment_column}' not found")

        control_mask = adata.obs[treatment_column] == control_value

        control_adata = adata[control_mask]
        treated_adata = adata[~control_mask]

        # Get treatments
        treatments = treated_adata.obs[treatment_column].unique().tolist()

        return {
            "control": {
                "n_cells": control_adata.n_obs,
                "expression": control_adata.X,
                "obs": control_adata.obs.to_dict(),
            },
            "treatments": {
                t: {
                    "n_cells": (adata.obs[treatment_column] == t).sum(),
                }
                for t in treatments
            },
            "genes": adata.var_names.tolist(),
            "n_genes": adata.n_vars,
        }

    def cluster_cells(
        self,
        h5ad_path: Path | str,
        n_neighbors: int = 15,
        n_pcs: int = 50,
        resolution: float = 1.0,
    ) -> pd.DataFrame:
        """
        Perform clustering on cells.

        Args:
            h5ad_path: Path to h5ad file
            n_neighbors: Number of neighbors
            n_pcs: Number of principal components
            resolution: Clustering resolution

        Returns:
            DataFrame with cluster assignments
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy required for clustering")

        adata = self.load(h5ad_path)

        # Basic preprocessing if not done
        if "pca" not in adata.obsm:
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            sc.tl.pca(adata, n_comps=n_pcs)

        # Clustering
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.leiden(adata, resolution=resolution)

        return adata.obs[["leiden"]]

    def differential_expression(
        self,
        h5ad_path: Path | str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "wilcoxon",
    ) -> pd.DataFrame:
        """
        Perform differential expression analysis.

        Args:
            h5ad_path: Path to h5ad file
            groupby: Column to group by
            group1: First group
            group2: Second group (or "rest")
            method: Statistical method

        Returns:
            DataFrame with DE results
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy required for DE analysis")

        adata = self.load(h5ad_path)

        # Subset if comparing specific groups
        if group2 != "rest":
            adata = adata[adata.obs[groupby].isin([group1, group2])]

        # Run DE
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            groups=[group1],
            reference=group2,
            method=method,
        )

        # Extract results
        results = sc.get.rank_genes_groups_df(adata, group=group1)

        return results


def analyze_h5ad_file(h5ad_path: str, **kwargs) -> dict:
    """
    Analyze an h5ad single-cell data file.

    Args:
        h5ad_path: Path to h5ad file

    Returns:
        Dictionary with analysis results
    """
    handler = H5ADHandler()
    summary = handler.summarize(h5ad_path)

    # Format cell type counts
    cell_type_str = ", ".join(
        f"{ct}: {count}" for ct, count in
        list(summary.cell_type_counts.items())[:5]
    )
    if len(summary.cell_type_counts) > 5:
        cell_type_str += f", ... ({len(summary.cell_type_counts)} total types)"

    return {
        "summary": (
            f"Loaded h5ad: {summary.n_cells} cells, {summary.n_genes} genes. "
            f"Cell types: {cell_type_str}"
        ),
        "n_cells": summary.n_cells,
        "n_genes": summary.n_genes,
        "cell_types": summary.cell_types,
        "cell_type_counts": summary.cell_type_counts,
        "obs_columns": summary.obs_columns,
        "var_columns": summary.var_columns,
        "has_raw": summary.has_raw,
        "has_spatial": summary.has_spatial,
    }


def extract_gene_expression(
    h5ad_path: str,
    genes: str,
    **kwargs,
) -> dict:
    """
    Extract expression for specific genes.

    Args:
        h5ad_path: Path to h5ad file
        genes: Comma-separated gene names

    Returns:
        Dictionary with expression data
    """
    handler = H5ADHandler()
    gene_list = [g.strip() for g in genes.split(",") if g.strip()]

    expr_df = handler.extract_expression(h5ad_path, gene_list)

    if expr_df.empty:
        return {
            "summary": "No genes found",
            "expression": {},
        }

    # Calculate statistics
    stats = {}
    for gene in expr_df.columns:
        stats[gene] = {
            "mean": float(expr_df[gene].mean()),
            "median": float(expr_df[gene].median()),
            "std": float(expr_df[gene].std()),
            "fraction_expressed": float((expr_df[gene] > 0).mean()),
        }

    return {
        "summary": f"Extracted expression for {len(expr_df.columns)} genes across {len(expr_df)} cells",
        "genes_found": expr_df.columns.tolist(),
        "n_cells": len(expr_df),
        "statistics": stats,
    }