# CellType-Agent Phase 1: Foundation Implementation

> **Comprehensive implementation of knowledge graph, ADMET prediction, GPU infrastructure, and session logging for the In-Silico Scientist vision.**

## Overview

Phase 1 establishes the foundation for transforming CellType-Agent from a tool-calling orchestrator into a generative, hallucination-resistant research system. This implementation follows the revised timeline and priorities from the Deep Research Analysis.

## Key Components

### 1. Knowledge Graph (DRKG + Neo4j)

**Goal:** Reduce pathway hallucinations from ~15% to <3%

```
src/ct/knowledge_graph/
├── __init__.py
├── drkg_loader.py      # Download and ingest DRKG
├── neo4j_client.py     # High-level Neo4j interface
├── graphrag_queries.py # 20+ query templates
└── text_to_cypher.py   # LLM-powered translation
```

**Quick Start:**
```python
from ct.knowledge_graph import DRKGLoader, Neo4jClient

# Load DRKG into Neo4j
loader = DRKGLoader(neo4j_uri="bolt://localhost:7687")
stats = loader.load()  # ~97K entities, 4.4M relations

# Query the knowledge graph
client = Neo4jClient()
targets = client.get_drug_targets("imatinib")
diseases = client.get_gene_diseases("TP53")
```

**Query Templates (20+):**
| Template | Use Case |
|----------|----------|
| `drug_targets` | Get all protein targets for a drug |
| `drug_diseases` | Get diseases treated by a drug |
| `drug_side_effects` | Get side effects of a drug |
| `gene_diseases` | Get diseases associated with a gene |
| `gene_pathways` | Get pathways a gene participates in |
| `pathway_drugs` | Get drugs targeting a pathway |
| `similar_drugs` | Find drugs sharing targets |
| `combination_targets` | Find combination target candidates |

### 2. ADMET Prediction (41 Endpoints)

**Goal:** Fast, comprehensive ADMET screening in <2 seconds

```
src/ct/admet/
├── __init__.py
├── endpoints.py        # 41 endpoint definitions
└── predictor.py        # GNN-based prediction
```

**Endpoint Categories:**
- **Absorption (8):** Caco-2, PAMPA, P-gp, Bioavailability, HIA
- **Distribution (5):** BBB, PPB, Vdss, Fraction unbound
- **Metabolism (10):** CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2 (substrate/inhibitor)
- **Excretion (4):** Clearance, Half-life, Tmax
- **Toxicity (14):** hERG, DILI, Ames, Carcinogenicity, Skin sensitization, etc.

**Quick Start:**
```python
from ct.admet import ADMETPredictor

predictor = ADMETPredictor()
result = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin

print(result.predictions["herg_inhibitor"])  # hERG risk
print(result.predictions["bbb_permeability"])  # BBB penetration
print(result.flags)  # List of issues
print(result.confidence)  # Overall confidence
```

**Critical Endpoints (always checked):**
1. hERG inhibition
2. Ames mutagenicity
3. Drug-induced liver injury (DILI)
4. BBB permeability
5. CYP3A4 inhibition
6. CYP2D6 inhibition
7. CYP2C9 inhibition
8. Oral bioavailability >20%

### 3. GPU Infrastructure

**Goal:** Efficient batch inference for virtual screening

```
src/ct/gpu_infrastructure/
├── __init__.py
├── resource_manager.py  # GPU detection and allocation
├── batch_processor.py   # Batch processing engine
└── boltz2_optimizer.py  # Boltz-2 virtual screening
```

**Realistic Performance:**
| Task | Time | Notes |
|------|------|-------|
| Single prediction | 15-30 sec | Single protein-ligand pair |
| 1K ligands | 25-45 min | Batch size 32, single A100 |
| 10K ligands | 4-6 hours | Single A100 |
| 100K ligands | 40-60 hours | Single A100; use multi-GPU |

**Quick Start:**
```python
from ct.gpu_infrastructure import Boltz2Optimizer

optimizer = Boltz2Optimizer()

# Single prediction
result = optimizer.predict_affinity(
    protein_sequence="MKT...",
    ligand_smiles="CCO",
)

# Virtual screening
screening = optimizer.virtual_screen(
    protein_sequence="MKT...",
    ligand_smiles_list=["CCO", "CCN", ...],
    top_k=100,
)
```

### 4. Session Logging (RLEF Foundation)

**Goal:** Collect 15K+ high-quality traces for LoRA training

```
src/ct/session_logging/
├── __init__.py
├── logger.py            # Session lifecycle logging
├── trace_store.py       # SQLite persistence
└── feedback_collector.py # User feedback collection
```

**Quick Start:**
```python
from ct.session_logging import SessionLogger

logger = SessionLogger()
logger.start_session("What drugs target KRAS?")
logger.log_tool_call("chembl.search", {"query": "KRAS"}, result)
logger.end_session("Found 50 approved KRAS inhibitors")

# Add feedback
logger.add_feedback(rating=4, outcome="validated")
```

**Feedback Options:**
- Rating: 1-5 stars
- Outcome: `validated`, `partially_validated`, `refuted`, `inconclusive`
- Text feedback

## Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x (for knowledge graph)
- NVIDIA GPU with 24GB+ VRAM (for Boltz-2)
- CUDA 12.x

### Quick Install

```bash
# Clone repository
git clone https://github.com/SamerKharboush/DD.git
cd DD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements-phase1.txt

# Install Neo4j (Docker recommended)
docker run -d -p 7474:7474 -p 7687:7687 neo4j:5

# Load DRKG
python -c "from ct.knowledge_graph import DRKGLoader; DRKGLoader().load()"
```

### Verify Installation

```bash
# Run tests
pytest tests/test_phase1.py -v

# Check GPU
python -c "from ct.gpu_infrastructure import GPUResourceManager; print(GPUResourceManager().get_summary())"

# Check session stats
python -c "from ct.session_logging import SessionLogger; print(SessionLogger().get_stats())"
```

## Tool Reference

### Knowledge Graph Tools

| Tool | Description |
|------|-------------|
| `knowledge.search_entities` | Search for genes, drugs, diseases, pathways |
| `knowledge.get_drug_targets` | Get protein targets for a drug |
| `knowledge.get_gene_diseases` | Get diseases associated with a gene |
| `knowledge.find_path` | Find shortest path between entities |
| `knowledge.graphrag_query` | Natural language query on knowledge graph |

### ADMET Tools

| Tool | Description |
|------|-------------|
| `admet.predict` | Predict 41 ADMET endpoints |
| `admet.batch_predict` | Batch prediction for multiple compounds |
| `admet.compare_compounds` | Compare and rank compounds by ADMET |

### Structure Tools

| Tool | Description |
|------|-------------|
| `boltz2.predict_affinity` | Single protein-ligand affinity prediction |
| `boltz2.virtual_screen` | Virtual screen library against target |

### Session Tools

| Tool | Description |
|------|-------------|
| `session.feedback` | Provide feedback for RLEF |
| `session.stats` | View session statistics |

## Configuration

Add to `~/.ct/config.json`:

```json
{
  "knowledge.neo4j_uri": "bolt://localhost:7687",
  "knowledge.neo4j_user": "neo4j",
  "knowledge.neo4j_password": "password",

  "admet.backend": "auto",
  "admet.cache_predictions": true,

  "gpu.min_vram_gb": 20,

  "session.min_quality_score": 0.5
}
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| DRKG entities loaded | 97K+ | ✅ |
| DRKG relations loaded | 4.4M+ | ✅ |
| ADMET prediction time | <2 sec | ✅ |
| Boltz-2 single prediction | <30 sec | ✅ |
| Session log retention | Persistent | ✅ |
| Hallucination reduction | <10% | 🔄 Testing |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_phase1.py::TestADMETPredictor -v

# Run with coverage
pytest tests/ --cov=src/ct --cov-report=html
```

## Next Steps (Phase 2)

1. **BoltzGen Integration** - De novo binder design
2. **ESM3 API** - Conditional protein generation
3. **Text-to-Cypher LLM** - Complex query translation
4. **Validation Pipeline** - ESM-IF, Aggrescan, NetMHCpan

## References

- [DRKG: Drug Repurposing Knowledge Graph](https://github.com/gnn4dr/DRKG)
- [Boltz-2: Structure and Affinity Prediction](https://github.com/jwohlwend/boltz)
- [ADMET-AI: 41-endpoint Prediction](https://github.com/swansonk14/admet_ai)
- [Neo4j Graph Database](https://neo4j.com/)

## License

MIT License - See [LICENSE](LICENSE)

---

Built by [CellType Inc.](https://celltype.com) | Phase 1 Implementation | March 2026