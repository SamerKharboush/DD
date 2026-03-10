"""
ADMET endpoint definitions and priority levels.

41 endpoints across 5 categories:
- Absorption (8 endpoints)
- Distribution (5 endpoints)
- Metabolism (10 endpoints)
- Excretion (4 endpoints)
- Toxicity (14 endpoints)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EndpointCategory(Enum):
    """ADMET endpoint categories."""
    ABSORPTION = "absorption"
    DISTRIBUTION = "distribution"
    METABOLISM = "metabolism"
    EXCRETION = "excretion"
    TOXICITY = "toxicity"


@dataclass
class ADMETEndpoint:
    """Definition of an ADMET prediction endpoint."""
    name: str
    display_name: str
    category: EndpointCategory
    description: str
    units: Optional[str]
    optimal_range: Optional[tuple] = None
    clinical_relevance: str = ""
    priority: int = 2  # 1=critical, 2=important, 3=optional
    model_type: str = "classification"  # classification or regression


# Complete list of 41 ADMET endpoints
ADMET_ENDPOINTS = {
    # ============================================================
    # ABSORPTION (8 endpoints)
    # ============================================================
    "caco2_permeability": ADMETEndpoint(
        name="caco2_permeability",
        display_name="Caco-2 Permeability",
        category=EndpointCategory.ABSORPTION,
        description="Predicts apparent permeability across Caco-2 cell monolayers (cm/s)",
        units="cm/s",
        optimal_range=(1e-6, 1e-4),
        clinical_relevance="Indicates intestinal absorption potential",
        priority=2,
        model_type="regression",
    ),
    "pamap_permeability": ADMETEndpoint(
        name="pamap_permeability",
        display_name="PAMPA Permeability",
        category=EndpointCategory.ABSORPTION,
        description="Parallel artificial membrane permeability assay",
        units="cm/s",
        priority=3,
        model_type="regression",
    ),
    "pgp_inhibitor": ADMETEndpoint(
        name="pgp_inhibitor",
        display_name="P-gp Inhibition",
        category=EndpointCategory.ABSORPTION,
        description="Whether compound inhibits P-glycoprotein efflux pump",
        units=None,
        clinical_relevance="May cause drug-drug interactions",
        priority=2,
        model_type="classification",
    ),
    "pgp_substrate": ADMETEndpoint(
        name="pgp_substrate",
        display_name="P-gp Substrate",
        category=EndpointCategory.ABSORPTION,
        description="Whether compound is a P-glycoprotein substrate",
        units=None,
        clinical_relevance="Affects oral bioavailability and BBB penetration",
        priority=2,
        model_type="classification",
    ),
    "bioavailability_f20": ADMETEndpoint(
        name="bioavailability_f20",
        display_name="Oral Bioavailability >20%",
        category=EndpointCategory.ABSORPTION,
        description="Probability of oral bioavailability >20%",
        units=None,
        clinical_relevance="Key for oral drug development",
        priority=1,
        model_type="classification",
    ),
    "bioavailability_f30": ADMETEndpoint(
        name="bioavailability_f30",
        display_name="Oral Bioavailability >30%",
        category=EndpointCategory.ABSORPTION,
        description="Probability of oral bioavailability >30%",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "hia": ADMETEndpoint(
        name="hia",
        display_name="Human Intestinal Absorption",
        category=EndpointCategory.ABSORPTION,
        description="Fraction absorbed from GI tract",
        units="%",
        optimal_range=(70, 100),
        clinical_relevance="Critical for oral drugs",
        priority=1,
        model_type="regression",
    ),
    "hia_class": ADMETEndpoint(
        name="hia_class",
        display_name="HIA Classification",
        category=EndpointCategory.ABSORPTION,
        description="High/low intestinal absorption classification",
        units=None,
        priority=2,
        model_type="classification",
    ),

    # ============================================================
    # DISTRIBUTION (5 endpoints)
    # ============================================================
    "bbb_permeability": ADMETEndpoint(
        name="bbb_permeability",
        display_name="BBB Permeability",
        category=EndpointCategory.DISTRIBUTION,
        description="Blood-brain barrier penetration potential",
        units=None,
        clinical_relevance="Essential for CNS drugs, avoid for non-CNS",
        priority=1,
        model_type="classification",
    ),
    "bbb_permeant": ADMETEndpoint(
        name="bbb_permeant",
        display_name="BBB Permeant",
        category=EndpointCategory.DISTRIBUTION,
        description="Whether compound crosses BBB",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "ppb": ADMETEndpoint(
        name="ppb",
        display_name="Plasma Protein Binding",
        category=EndpointCategory.DISTRIBUTION,
        description="Fraction bound to plasma proteins",
        units="%",
        optimal_range=(0, 95),
        clinical_relevance="Affects free drug concentration",
        priority=2,
        model_type="regression",
    ),
    "vdss": ADMETEndpoint(
        name="vdss",
        display_name="Volume of Distribution",
        category=EndpointCategory.DISTRIBUTION,
        description="Steady-state volume of distribution",
        units="L/kg",
        clinical_relevance="Indicates tissue distribution",
        priority=2,
        model_type="regression",
    ),
    "fraction_unbound": ADMETEndpoint(
        name="fraction_unbound",
        display_name="Fraction Unbound",
        category=EndpointCategory.DISTRIBUTION,
        description="Unbound fraction in plasma",
        units=None,
        optimal_range=(0.05, 0.5),
        priority=3,
        model_type="regression",
    ),

    # ============================================================
    # METABOLISM (10 endpoints)
    # ============================================================
    "cyp3a4_inhibitor": ADMETEndpoint(
        name="cyp3a4_inhibitor",
        display_name="CYP3A4 Inhibition",
        category=EndpointCategory.METABOLISM,
        description="Whether compound inhibits CYP3A4",
        units=None,
        clinical_relevance="Major drug-drug interaction liability",
        priority=1,
        model_type="classification",
    ),
    "cyp3a4_substrate": ADMETEndpoint(
        name="cyp3a4_substrate",
        display_name="CYP3A4 Substrate",
        category=EndpointCategory.METABOLISM,
        description="Whether compound is metabolized by CYP3A4",
        units=None,
        clinical_relevance="Most important metabolizing enzyme",
        priority=1,
        model_type="classification",
    ),
    "cyp2d6_inhibitor": ADMETEndpoint(
        name="cyp2d6_inhibitor",
        display_name="CYP2D6 Inhibition",
        category=EndpointCategory.METABOLISM,
        description="Whether compound inhibits CYP2D6",
        units=None,
        clinical_relevance="Common polymorphic enzyme",
        priority=1,
        model_type="classification",
    ),
    "cyp2d6_substrate": ADMETEndpoint(
        name="cyp2d6_substrate",
        display_name="CYP2D6 Substrate",
        category=EndpointCategory.METABOLISM,
        description="Whether compound is metabolized by CYP2D6",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "cyp2c9_inhibitor": ADMETEndpoint(
        name="cyp2c9_inhibitor",
        display_name="CYP2C9 Inhibition",
        category=EndpointCategory.METABOLISM,
        description="Whether compound inhibits CYP2C9",
        units=None,
        clinical_relevance="Important for warfarin interactions",
        priority=1,
        model_type="classification",
    ),
    "cyp2c9_substrate": ADMETEndpoint(
        name="cyp2c9_substrate",
        display_name="CYP2C9 Substrate",
        category=EndpointCategory.METABOLISM,
        description="Whether compound is metabolized by CYP2C9",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "cyp2c19_inhibitor": ADMETEndpoint(
        name="cyp2c19_inhibitor",
        display_name="CYP2C19 Inhibition",
        category=EndpointCategory.METABOLISM,
        description="Whether compound inhibits CYP2C19",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "cyp2c19_substrate": ADMETEndpoint(
        name="cyp2c19_substrate",
        display_name="CYP2C19 Substrate",
        category=EndpointCategory.METABOLISM,
        description="Whether compound is metabolized by CYP2C19",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "cyp1a2_inhibitor": ADMETEndpoint(
        name="cyp1a2_inhibitor",
        display_name="CYP1A2 Inhibition",
        category=EndpointCategory.METABOLISM,
        description="Whether compound inhibits CYP1A2",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "cyp1a2_substrate": ADMETEndpoint(
        name="cyp1a2_substrate",
        display_name="CYP1A2 Substrate",
        category=EndpointCategory.METABOLISM,
        description="Whether compound is metabolized by CYP1A2",
        units=None,
        priority=3,
        model_type="classification",
    ),

    # ============================================================
    # EXCRETION (4 endpoints)
    # ============================================================
    "clearance_hepatic": ADMETEndpoint(
        name="clearance_hepatic",
        display_name="Hepatic Clearance",
        category=EndpointCategory.EXCRETION,
        description="Liver clearance rate",
        units="mL/min/kg",
        clinical_relevance="Affects dosing frequency",
        priority=2,
        model_type="regression",
    ),
    "clearance_microsomal": ADMETEndpoint(
        name="clearance_microsomal",
        display_name="Microsomal Clearance",
        category=EndpointCategory.EXCRETION,
        description="Intrinsic clearance in liver microsomes",
        units="mL/min/kg",
        priority=3,
        model_type="regression",
    ),
    "half_life": ADMETEndpoint(
        name="half_life",
        display_name="Half-life",
        category=EndpointCategory.EXCRETION,
        description="Elimination half-life",
        units="hours",
        optimal_range=(4, 24),
        clinical_relevance="Determines dosing frequency",
        priority=2,
        model_type="regression",
    ),
    "tmax": ADMETEndpoint(
        name="tmax",
        display_name="Tmax",
        category=EndpointCategory.EXCRETION,
        description="Time to maximum concentration",
        units="hours",
        priority=3,
        model_type="regression",
    ),

    # ============================================================
    # TOXICITY (14 endpoints)
    # ============================================================
    "herg_inhibitor": ADMETEndpoint(
        name="herg_inhibitor",
        display_name="hERG Inhibition",
        category=EndpointCategory.TOXICITY,
        description="Whether compound inhibits hERG potassium channel",
        units=None,
        clinical_relevance="CRITICAL - Can cause fatal cardiac arrhythmias",
        priority=1,
        model_type="classification",
    ),
    "herg_ic50": ADMETEndpoint(
        name="herg_ic50",
        display_name="hERG IC50",
        category=EndpointCategory.TOXICITY,
        description="IC50 for hERG channel inhibition",
        units="μM",
        optimal_range=(10, 100),
        clinical_relevance="CRITICAL - Safety margin assessment",
        priority=1,
        model_type="regression",
    ),
    "ames_mutagenicity": ADMETEndpoint(
        name="ames_mutagenicity",
        display_name="Ames Mutagenicity",
        category=EndpointCategory.TOXICITY,
        description="Whether compound is mutagenic in Ames test",
        units=None,
        clinical_relevance="CRITICAL - Genotoxicity liability",
        priority=1,
        model_type="classification",
    ),
    "dili": ADMETEndpoint(
        name="dili",
        display_name="Drug-Induced Liver Injury",
        category=EndpointCategory.TOXICITY,
        description="Risk of causing liver injury",
        units=None,
        clinical_relevance="CRITICAL - Major cause of drug failure",
        priority=1,
        model_type="classification",
    ),
    "skin_sensitization": ADMETEndpoint(
        name="skin_sensitization",
        display_name="Skin Sensitization",
        category=EndpointCategory.TOXICITY,
        description="Whether compound causes skin sensitization",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "respiratory_toxicity": ADMETEndpoint(
        name="respiratory_toxicity",
        display_name="Respiratory Toxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of respiratory toxicity",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "carcinogenicity": ADMETEndpoint(
        name="carcinogenicity",
        display_name="Carcinogenicity",
        category=EndpointCategory.TOXICITY,
        description="Whether compound is carcinogenic",
        units=None,
        clinical_relevance="Major regulatory concern",
        priority=1,
        model_type="classification",
    ),
    "mitochondrial_toxicity": ADMETEndpoint(
        name="mitochondrial_toxicity",
        display_name="Mitochondrial Toxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of mitochondrial dysfunction",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "nephrotoxicity": ADMETEndpoint(
        name="nephrotoxicity",
        display_name="Nephrotoxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of kidney injury",
        units=None,
        clinical_relevance="Important for chronic dosing",
        priority=2,
        model_type="classification",
    ),
    "ototoxicity": ADMETEndpoint(
        name="ototoxicity",
        display_name="Ototoxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of hearing damage",
        units=None,
        priority=3,
        model_type="classification",
    ),
    "cardiotoxicity": ADMETEndpoint(
        name="cardiotoxicity",
        display_name="Cardiotoxicity",
        category=EndpointCategory.TOXICITY,
        description="General cardiac toxicity risk",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "neurotoxicity": ADMETEndpoint(
        name="neurotoxicity",
        display_name="Neurotoxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of neurological damage",
        units=None,
        priority=2,
        model_type="classification",
    ),
    "hepatotoxicity": ADMETEndpoint(
        name="hepatotoxicity",
        display_name="Hepatotoxicity",
        category=EndpointCategory.TOXICITY,
        description="Risk of liver damage",
        units=None,
        clinical_relevance="Major safety concern",
        priority=1,
        model_type="classification",
    ),
    "ld50": ADMETEndpoint(
        name="ld50",
        display_name="LD50",
        category=EndpointCategory.TOXICITY,
        description="Median lethal dose",
        units="mg/kg",
        priority=3,
        model_type="regression",
    ),
}

# Critical endpoints that must be checked for every compound
CRITICAL_ENDPOINTS = [
    "herg_inhibitor",
    "ames_mutagenicity",
    "dili",
    "bbb_permeability",
    "cyp3a4_inhibitor",
    "cyp2d6_inhibitor",
    "cyp2c9_inhibitor",
    "bioavailability_f20",
]

# Category summary
ENDPOINT_COUNTS = {
    EndpointCategory.ABSORPTION: 8,
    EndpointCategory.DISTRIBUTION: 5,
    EndpointCategory.METABOLISM: 10,
    EndpointCategory.EXCRETION: 4,
    EndpointCategory.TOXICITY: 14,
}


def get_endpoints_by_category(category: EndpointCategory) -> dict[str, ADMETEndpoint]:
    """Get all endpoints in a specific category."""
    return {
        name: endpoint
        for name, endpoint in ADMET_ENDPOINTS.items()
        if endpoint.category == category
    }


def get_endpoints_by_priority(priority: int) -> dict[str, ADMETEndpoint]:
    """Get all endpoints with a specific priority level."""
    return {
        name: endpoint
        for name, endpoint in ADMET_ENDPOINTS.items()
        if endpoint.priority == priority
    }