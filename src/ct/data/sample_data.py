"""
Sample Data for Testing and Demos.

Provides sample compounds, targets, and queries for development.
"""

from typing import Optional


# Sample compounds with known properties
SAMPLE_COMPOUNDS = {
    "sotorasib": {
        "name": "Sotorasib (AMG 510)",
        "smiles": "CC(C)C[C@H](N)C(=O)Nc1ccc(F)cc1C(=O)Nc2nccc(N3CCN(C)CC3)n2",
        "target": "KRAS G12C",
        "indication": "Non-small cell lung cancer",
        "status": "FDA Approved",
        "mw": 560.6,
    },
    "adagrasib": {
        "name": "Adagrasib (MRTX849)",
        "smiles": "CC(C)C[C@H](N)C(=O)Nc1ccc(F)cc1NC(=O)c2ccccc2Cl",
        "target": "KRAS G12C",
        "indication": "Non-small cell lung cancer",
        "status": "FDA Approved",
        "mw": 604.5,
    },
    "imatinib": {
        "name": "Imatinib (Gleevec)",
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "target": "BCR-ABL",
        "indication": "Chronic myeloid leukemia",
        "status": "FDA Approved",
        "mw": 493.6,
    },
    "aspirin": {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "target": "COX-1/COX-2",
        "indication": "Pain, inflammation",
        "status": "FDA Approved",
        "mw": 180.2,
    },
    "metformin": {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "target": "AMPK",
        "indication": "Type 2 diabetes",
        "status": "FDA Approved",
        "mw": 129.2,
    },
    "caffeine": {
        "name": "Caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "target": "Adenosine receptor",
        "indication": "Stimulant",
        "status": "FDA Approved",
        "mw": 194.2,
    },
    "ethanol": {
        "name": "Ethanol",
        "smiles": "CCO",
        "target": "GABA-A receptor",
        "indication": "Solvent, recreational",
        "status": "Not a drug",
        "mw": 46.1,
    },
    "paracetamol": {
        "name": "Paracetamol (Acetaminophen)",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "target": "COX enzymes",
        "indication": "Pain, fever",
        "status": "FDA Approved",
        "mw": 151.2,
    },
}


# Sample targets
SAMPLE_TARGETS = {
    "KRAS": {
        "name": "KRAS",
        "full_name": "Kirsten rat sarcoma viral oncogene",
        "uniprot": "P01116",
        "type": "GTPase",
        "disease_associations": ["Cancer", "RASopathy"],
        "drugs": ["Sotorasib", "Adagrasib"],
    },
    "EGFR": {
        "name": "EGFR",
        "full_name": "Epidermal growth factor receptor",
        "uniprot": "P00533",
        "type": "Receptor tyrosine kinase",
        "disease_associations": ["Non-small cell lung cancer", "Colorectal cancer"],
        "drugs": ["Erlotinib", "Gefitinib", "Osimertinib"],
    },
    "BCR-ABL": {
        "name": "BCR-ABL",
        "full_name": "Breakpoint cluster region-Abelson fusion protein",
        "uniprot": "P00519",
        "type": "Tyrosine kinase",
        "disease_associations": ["Chronic myeloid leukemia"],
        "drugs": ["Imatinib", "Dasatinib", "Nilotinib"],
    },
    "BRAF": {
        "name": "BRAF",
        "full_name": "B-Raf proto-oncogene, serine/threonine kinase",
        "uniprot": "P15056",
        "type": "Serine/threonine kinase",
        "disease_associations": ["Melanoma", "Colorectal cancer"],
        "drugs": ["Vemurafenib", "Dabrafenib"],
    },
    "ALK": {
        "name": "ALK",
        "full_name": "Anaplastic lymphoma kinase",
        "uniprot": "Q9UM73",
        "type": "Receptor tyrosine kinase",
        "disease_associations": ["Non-small cell lung cancer"],
        "drugs": ["Crizotinib", "Alectinib", "Lorlatinib"],
    },
    "PD-1": {
        "name": "PD-1",
        "full_name": "Programmed cell death protein 1",
        "uniprot": "Q15116",
        "type": "Immune checkpoint receptor",
        "disease_associations": ["Various cancers"],
        "drugs": ["Pembrolizumab", "Nivolumab"],
    },
}


# Sample queries for testing
SAMPLE_QUERIES = [
    # Simple lookups
    "What drugs target KRAS?",
    "What is the mechanism of action of sotorasib?",
    "List FDA-approved EGFR inhibitors.",

    # Comparative
    "Compare sotorasib and adagrasib.",
    "What are the differences between imatinib and dasatinib?",

    # Analysis
    "Predict ADMET properties for aspirin.",
    "Is this compound drug-like: Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "What are the potential off-targets of sotorasib?",

    # Design
    "Design a KRAS G12C inhibitor with improved solubility.",
    "Suggest modifications to improve the drug-likeness of this compound.",
    "Generate analogs of imatinib with reduced cardiotoxicity.",

    # Multi-agent
    "Analyze the safety profile of paracetamol.",
    "Evaluate the therapeutic potential of targeting KRAS G12D.",
    "What biomarkers predict response to EGFR inhibitors?",

    # Complex
    "Run a DMTA cycle for KRAS G12C.",
    "Find drug combinations for melanoma.",
    "What are the resistance mechanisms to osimertinib?",
]


def get_sample_compounds(limit: Optional[int] = None) -> list[dict]:
    """
    Get sample compounds for testing.

    Args:
        limit: Maximum number to return

    Returns:
        List of compound dictionaries
    """
    compounds = list(SAMPLE_COMPOUNDS.values())
    if limit:
        compounds = compounds[:limit]
    return compounds


def get_sample_targets(limit: Optional[int] = None) -> list[dict]:
    """
    Get sample targets for testing.

    Args:
        limit: Maximum number to return

    Returns:
        List of target dictionaries
    """
    targets = list(SAMPLE_TARGETS.values())
    if limit:
        targets = targets[:limit]
    return targets


def get_sample_queries(limit: Optional[int] = None) -> list[str]:
    """
    Get sample queries for testing.

    Args:
        limit: Maximum number to return

    Returns:
        List of query strings
    """
    queries = SAMPLE_QUERIES.copy()
    if limit:
        queries = queries[:limit]
    return queries


def get_test_smiles() -> list[str]:
    """Get SMILES strings for testing."""
    return [c["smiles"] for c in SAMPLE_COMPOUNDS.values()]