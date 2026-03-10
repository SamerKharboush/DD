# 🧬 CellType-Agent — Next-Generation AI Enhancement Master Plan

> **Deep Research Report & Strategic Implementation Roadmap**
> Compiled March 2026 | Based on SOTA Research Through 2025–2026
> Covers: Boltz-2 · BoltzGen · ESM3 · scGPT · Robin · GraphRAG · DRKG · Agentic Bioinformatics

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Structural Biology Revolution: The Boltz Family](#2-structural-biology-revolution-the-boltz-family)
3. [Protein Language Models: ESM3 & ESM C](#3-protein-language-models-esm3--esm-c)
4. [Single-Cell Foundation Models: scGPT, Geneformer & Arc](#4-single-cell-foundation-models-scgpt-geneformer--arc)
5. [Multi-Agent Orchestration: The Robin Paradigm](#5-multi-agent-orchestration-the-robin-paradigm)
6. [GraphRAG: Hallucination-Free Biological Reasoning](#6-graphrag-hallucination-free-biological-reasoning)
7. [Graph Neural Networks for Polypharmacology & ADMET](#7-graph-neural-networks-for-polypharmacology--admet)
8. [Enterprise MLOps: Local LLMs & Privacy-Preserving Inference](#8-enterprise-mlops-local-llms--privacy-preserving-inference)
9. [Multi-Modal Inputs: Vision, Structure & Omics](#9-multi-modal-inputs-vision-structure--omics)
10. [Reinforcement Learning from Experimental Feedback (RLEF)](#10-reinforcement-learning-from-experimental-feedback-rlef)
11. [Implementation Roadmap: 9-Month Plan](#11-implementation-roadmap-9-month-plan)
12. [Key Open-Source Projects to Integrate](#12-key-open-source-projects-to-integrate)
13. [Projected Performance Targets](#13-projected-performance-targets)
14. [Conclusion: The In-Silico Scientist Vision](#14-conclusion-the-in-silico-scientist-vision)

---

## 1. Executive Summary

CellType-Agent currently stands as the top-performing autonomous biological research agent on BixBench-Verified-50, achieving **90% accuracy** — more than 24 points ahead of Claude Code alone. It integrates 190+ drug discovery tools, 30+ database APIs, and Claude-powered multi-step reasoning to serve computational biologists and drug discovery researchers.

This report synthesizes findings from deep web research conducted in early 2026, incorporating the most recent breakthroughs in the field, including the MIT/Recursion **Boltz-2** model (June 2025), **BoltzGen** for universal binder design (October 2025), **ESM3** from EvolutionaryScale (published in *Science*, January 2025), **scGPT** and the emerging field of single-cell foundation models, **Robin**'s landmark multi-agent drug discovery (May 2025), and **GraphRAG** architectures now being deployed in pharma companies including Merck, Bayer, and Lilly.

The transformation proposed here moves CellType-Agent from a highly capable tool-calling orchestrator into a fully generative, hallucination-resistant, multi-modal **In-Silico Scientist** — one that not only queries and analyzes, but designs, generates, critiques, and learns.

### Current State Benchmarks

| System | BixBench Accuracy | Paradigm |
|---|---|---|
| **CellType-Agent (Opus 4.6)** | **90.0%** | Agentic / Tool-Use |
| Phylo BiomniLab | 88.7% | Agentic / Tool-Use |
| Edison Analysis | 78.0% | Semi-Agentic |
| Claude Code (Opus 4.6) | 65.3% | Code-Gen |
| OpenAI Agents SDK (GPT 5.2) | 61.3% | Code-Gen |

---

## 2. Structural Biology Revolution: The Boltz Family

The most significant development in computational drug discovery in 2025 was the release of the open-source Boltz model family from MIT and Recursion. CellType-Agent v0.2.0 has already integrated Boltz-2 as a GPU-accelerated tool — this section explains how to dramatically deepen this integration.

### 2.1 Boltz-2: The Binding Affinity Breakthrough

Released June 6, 2025, **Boltz-2** is the first AI model to approach the accuracy of physics-based Free Energy Perturbation (FEP) methods for binding affinity prediction, while running **1,000× faster**. A single protein-ligand affinity prediction that previously required hours now takes 15–30 seconds on a single GPU.

> **Why It Matters:** Determining binding affinity is the single biggest bottleneck in drug discovery. Boltz-2 achieves 0.6 Pearson correlation on the gold-standard FEP+ benchmark — essentially eliminating the need for costly MD simulation in early-stage screening.

Key technical innovations in Boltz-2 that CellType-Agent should exploit:

- Joint structure + affinity prediction in a single model pass (no separate docking step)
- Affinity output: `affinity_probability_binary` (binder vs. decoy) for hit discovery; `affinity_pred_value` for SAR optimization
- Controllability: pocket constraints, distance constraints, multi-chain template integration, and method conditioning (X-ray/NMR/MD mode)
- Ensemble predictions including B-factor dynamics — giving a richer picture than static snapshots
- Outperformed all CASP16 affinity challenge participants without fine-tuning

### 2.2 BoltzGen: Universal Binder Design (Oct 2025)

**BoltzGen** (released October 26, 2025) is the generative extension of Boltz-2. It is currently the only fully open-source model capable of designing protein binders against **ANY** biomolecular target — proteins, nucleic acids, and small molecules alike — from scratch (de novo design).

> **Key Result:** BoltzGen achieved nanomolar-affinity binders for **66% of the 26 novel test targets** (with NO known binders in PDB). Wet-lab collaborators from UCSF, Harvard, and Parabilis Medicines validated results across antimicrobial, cancer therapy, and antibody design applications.

How to integrate BoltzGen into CellType-Agent:

- New tool: `/tool add boltzgen` — given a CRBN or other E3 ligase target sequence, BoltzGen designs a degrader binder de novo
- Feed the agent's target prioritization output directly into BoltzGen's design specification interface
- Use BoltzGen's constraint language to specify binding site, covalent bonds, linker constraints — enabling rational glue molecule design
- Output novel binders → validate with Boltz-2 affinity scoring → feed into RFdiffusion/ProteinMPNN pipeline for sequence optimization

**GitHub:** `github.com/HannesStark/boltzgen` | **License:** MIT | **Docker:** available

### 2.3 Proposed: Boltz End-to-End Pipeline Tool

| Step | Model / Tool | Output | Speed |
|---|---|---|---|
| Target identification | CellType-Agent tools | Protein target + pocket | Minutes |
| Virtual screening | Boltz-2 `affinity_probability_binary` | Top 100 binders from 100K library | ~30 min |
| Lead optimization | Boltz-2 `affinity_pred_value` | SAR ranking of analogs | Seconds/compound |
| De novo design | BoltzGen | Novel protein/peptide binders | ~1 hour batch |
| Sequence optimization | ProteinMPNN + ESM3 | Optimized sequences | Minutes |

---

## 3. Protein Language Models: ESM3 & ESM C

EvolutionaryScale's **ESM3**, published in *Science* (January 2025), is the first generative multimodal model that simultaneously reasons over protein sequence, 3D structure, and biological function. Trained on 2.78 billion protein sequences (771 billion tokens) with 1e24 FLOPS of compute, ESM3 at 98B parameters represents the largest and most capable protein foundation model ever built.

### 3.1 What Makes ESM3 Transformative

- **Promptable design:** scientists can specify partial sequence, partial structure, and/or functional keywords — ESM3 fills in the rest
- Demonstrated by generating **esmGFP** — a fluorescent protein at 58% identity from known proteins, equivalent to 500 million years of evolution
- **Geometric Attention** layer: encodes 3D spatial relationships between amino acids, allowing true structure-aware reasoning
- Fine-tuning with preference optimization improved generation success rate to **65.5%** on hard structural prompts
- **ESM C** (December 2024): A parallel family focused on representation learning, scaling to 6B parameters — delivers inference speedups while matching or exceeding ESM3 on many tasks

> **CellType-Agent Integration:** The agent should be able to prompt ESM3 with: *"Generate a CRBN neosubstrate degron that avoids the SALL4 off-target zinc finger interface."* This transforms target safety from a post-hoc filter into a generative constraint.

### 3.2 Concrete ESM3 Tool Additions

- `/tool add esm3-design` — conditional protein generation from sequence/structure/function prompts
- `/tool add esm3-score` — score existing sequences for stability, function keywords, off-target avoidance
- `/tool add esm3-mutate` — guided mutagenesis: propose point mutations to improve binding or reduce immunogenicity
- `/tool add esm-cambrian` — fast representation extraction for downstream GNN/ADMET models

**Access:** EvolutionaryScale Forge API (public beta), AWS, NVIDIA NIM. Weights for 1.4B open model on HuggingFace under non-commercial license. Full 7B/98B models via API.

---

## 4. Single-Cell Foundation Models: scGPT, Geneformer & Arc

Single-cell foundation models (scFMs) represent a convergence of big data biology and transformer architecture. They enable **in-silico perturbation experiments** at the single-cell level — predicting how a tumor microenvironment will respond to a given compound before any wet-lab work is done.

### 4.1 scGPT: The Perturbation Prediction Engine

scGPT (*Nature Methods*, February 2024) was pretrained on 33 million single-cell RNA-seq profiles. It excels at:

- **Perturbation prediction:** given drug treatment, predict transcriptional changes cell-type by cell-type
- **Reverse perturbation:** identify gene interventions that steer cells toward a desired state (e.g., resensitizing to immunotherapy)
- Cell type annotation and multi-omic integration
- Gene regulatory network (GRN) inference

> **⚠️ Key Caveat (2025 research):** *Genome Biology* 2025 showed scGPT and Geneformer underperform simpler baselines in zero-shot settings. Always use **fine-tuned variants** on task-specific data. Zero-shot should be paired with validation against scVI or Harmony baselines.

### 4.2 Arc Institute's Virtual Cell Model (June 2025)

The most important single-cell development of 2025 is the **Arc Institute's virtual cell model**, trained on 100M+ perturbation cells across 70 cell lines and 167M observational cells. It achieves **50%+ improvement** in perturbation effect discrimination on the Tahoe-100M benchmark and **2× accuracy** in identifying true differentially expressed genes.

- Architecture: State Embedding (SE) + State Transition (ST) modules with bidirectional attention over cell populations
- Trained on Perturb-seq data spanning 70 diverse cancer and normal cell lines
- Direct integration with DepMap and L1000 datasets — already present in CellType-Agent pipelines

### 4.3 Recommended Integration

| Tool | Use Case in CellType-Agent | Model |
|---|---|---|
| `/tool add virtual-cell` | In-silico drug perturbation on TME scRNA-seq | Arc Virtual Cell |
| `/tool add scgpt-perturb` | Predict immune-cold/hot transition from compound | scGPT fine-tuned |
| `/tool add geneformer-dosage` | Predict transcription factor activity at drug doses | Geneformer 12L |
| `/tool add grn-infer` | Build GRN for target pathway from scATAC+scRNA | GRNFormer/scGPT |

---

## 5. Multi-Agent Orchestration: The Robin Paradigm

The most compelling evidence that multi-agent systems work in drug discovery came in **May 2025**, when FutureHouse published **Robin** — the first AI system to fully automate the key intellectual steps of scientific discovery (hypothesis generation → experimental planning → data analysis → updated hypothesis) in a real drug discovery setting.

### 5.1 Robin: The Proof of Concept

Robin identified **ripasudil** (a ROCK inhibitor already used in ophthalmology) as a novel treatment for dry age-related macular degeneration (dAMD). It did so by:

- Using **Crow** (literature search agent) to review background and generate hypothesis
- Using **Falcon** (molecule evaluation agent) to screen 10 drug candidates
- Using **Finch** (data analysis agent) to interpret RNA-seq results and identify ABCA1 upregulation as the mechanism
- All hypotheses, experimental plans, data analyses, and figures in the main text were produced autonomously by Robin

> **Timeline:** From conceptualization to paper submission: **2.5 months**. With a 3-agent system. This is the benchmark CellType-Agent's multi-agent upgrade should aim to surpass.

### 5.2 Tippy: The DMTA Cycle Agent (2025)

**Tippy** (Artificial.com, July 2025) is the first production-ready multi-agent system for automating the Design-Make-Test-Analyze cycle in pharma. Its 5-agent architecture maps directly onto what CellType-Agent should implement:

| Tippy Agent | CellType-Agent Equivalent | Responsibility |
|---|---|---|
| Supervisor | `ct-orchestrator` | Break queries, assign tasks, merge results |
| Molecule Agent | `ct-chemist` | SAR, SMILES generation, synthesizability |
| Lab Agent | `ct-biologist` | Pathway analysis, cell line selection, assay design |
| Analysis Agent | `ct-statistician` | Dose-response, biomarker panel, enrichment |
| Report Agent | `ct-reporter` | HTML/PDF report synthesis, citation |
| Safety Guardrail | `ct-toxicologist` | SALL4 risk, pan-assay flags, ADMET |

### 5.3 Implementation Architecture for CellType-Agent

The existing `/agents N <query>` command is the foundation. The enhancement is to make each agent a **specialist** with its own system prompt and tool subset:

- `ct-chemist`: access to RDKit, ChEMBL, Boltz-2 affinity, BoltzGen design tools only
- `ct-biologist`: access to DepMap, L1000, scGPT, Reactome, PathFX tools only
- `ct-toxicologist` (critic): access to safety flagging, ADMET GNN, anti-target database — adversarial role
- `ct-orchestrator`: coordinates debate between agents, resolves conflicts, sends final plan to reporter
- Use Claude's existing Agent SDK sub-agent spawning — no need for AutoGen or CrewAI overhead
- Add **shared workspace**: JSON state object passed between agents with findings, confidence scores, and dissenting opinions

---

## 6. GraphRAG: Hallucination-Free Biological Reasoning

Standard vector-based RAG fails in biology because **relationships matter more than similarity**. A query like *"resistance biomarkers for degrader X"* requires traversing `Drug → inhibits → Protein → pathway → mutated gene → resistance phenotype` — a graph traversal problem, not a nearest-neighbor problem.

### 6.1 The GraphRAG Revolution in Pharma (2025)

GraphTalk 2025 (Neo4j's pharma conference, July 2025) confirmed that GraphRAG has moved from research to production:

- **Merck** deployed *Synaptix* — a knowledge graph + GraphRAG system for R&D knowledge management
- **Bayer** is using individual-level data integration with graph approaches
- **Lilly** launched a co-innovation lab with NVIDIA specifically for graph-based agentic drug discovery

### 6.2 Recommended Knowledge Graph Stack

| KG / Database | Entities | Use in CellType-Agent | Access |
|---|---|---|---|
| DRKG (Drug Repurposing KG) | 97K+ entities, 4.4M relations | Drug repurposing, polypharmacology | Open / GitHub |
| Hetionet | 11 entity types, 24 relation types | Pathway traversal, indication mapping | Neo4j online |
| PrimeKG | Proteins/diseases/drugs + synergy | Combination strategies | Open / HuggingFace |
| SPOKE | Comprehensive biomedical network | Target-disease-pathway-cell chains | API |
| DepMap as KG | Cell lines, genes, dependencies | Co-essentiality traversal | Local after `ct data pull` |

### 6.3 Implementation: Building CellType-Agent's BioKG

The most practical approach is to build a local Neo4j instance during `ct data pull` and expose it as a Cypher-query tool:

- **Phase 1:** Ingest DRKG into Neo4j. 97,000 nodes, 4.4 million edges. A TransE embedding model already pretrained on DRKG exists for GNN-based scoring.
- **Phase 2:** Extend with local DepMap data — create Cell Line nodes connected to Gene Dependency edges, enabling queries like *"which cancer cell lines are exclusively dependent on gene X?"*
- **Phase 3:** Add GraphRAG layer — convert graph neighborhood of a query node into a context string, inject into Claude's context window
- **Phase 4:** Use KG embeddings for drug repurposing scoring — rank existing drugs for new indications using path-based features (DWPC from Hetionet)

> **New Tool:** `/tool add biokg-query` — accepts Cypher-like natural language, translates to graph traversal, returns structured biological pathway context. Eliminates hallucinated pathway connections.

---

## 7. Graph Neural Networks for Polypharmacology & ADMET

Graph Neural Networks (GNNs) on molecular graphs are the current gold standard for compound-protein interaction (CPI) prediction and ADMET property prediction. Unlike fingerprint-based methods, GNNs learn the **chemical grammar** of molecular structure and generalize to novel scaffolds.

### 7.1 GNN ADMET Integration

- Implement **PyTorch Geometric** backend with pre-trained models from TDC (Therapeutics Data Commons)
- **ADMET-AI:** a 2024/2025 ensemble of GNN models trained on 41 ADMET endpoints — available as Python package, wraps directly as a tool
- **ChemProp** (MIT): message-passing neural network for molecular property prediction — excellent for SALL4 risk, hERG toxicity, BBB permeability
- **DeepPurpose:** drug-target affinity GNN using graph transformer — can screen against the full human proteome for off-target liability

> **New Tool:** `/tool add admet-gnn` — takes a SMILES string, returns 41 ADMET endpoints with confidence intervals. Runs in <1 second per compound.

### 7.2 Polypharmacology Network Analysis

For combination strategy queries (one of CellType-Agent's core use cases), GNNs can model synergy:

- **DrugComb / SynergyFinder:** use GNN trained on >900K drug combination data points to predict synergy scores (Bliss, Loewe, HSA)
- **MRGNN** (Multi-Relational GNN): models drug-drug-target interaction hypergraphs to predict emergent combination toxicity
- Build the **polypharmacology graph:** nodes = compounds + proteins; edges = known interactions, predicted interactions (from Boltz-2 + GNN); query paths = mechanism of synergy

---

## 8. Enterprise MLOps: Local LLMs & Privacy-Preserving Inference

For pharma and biotech teams, CellType-Agent's reliance on the Anthropic API creates IP exposure concerns. The enterprise roadmap requires a local deployment option with a biologically specialized model.

### 8.1 Recommended Local Model Stack

| Model | Parameters | Specialty | Deployment |
|---|---|---|---|
| BioMistral-7B | 7B | Biomedical QA, literature | vLLM / Ollama |
| Llama-3-70B-Instruct | 70B | General reasoning + tool use | vLLM on A100 |
| Meditron-70B | 70B | Clinical + biomedical | vLLM on H100 |
| OpenBioLLM-70B | 70B | BioNLP, pathway reasoning | vLLM / LMStudio |
| CellType Proprietary Model | TBD | celltype-agent tool calling | On-premise (CellType Inc.) |

### 8.2 LoRA Fine-Tuning on BixBench

The most impactful step for local model performance is training a LoRA adapter specifically on the BixBench-Verified-50 task distribution:

- Collect all CellType-Agent session traces (plan → tool calls → results → conclusion chains)
- Train a LoRA adapter (r=64, alpha=128) on Llama-3-70B to reproduce Claude's tool selection patterns
- Use **DPO** (Direct Preference Optimization) with researcher feedback as positive/negative pairs
- Target: **>80% BixBench accuracy** with Llama-3-70B + LoRA on a single A100 (vs. 90% with Opus 4.6)

### 8.3 Deployment: `celltype-agent-local` Docker

Release a containerized offline bundle:

- **Base:** CUDA 12.x + Python 3.11
- **LLM:** vLLM serving quantized (GPTQ/AWQ) local model on port 8080
- **Agent:** CellType-Agent CLI pointed at local LLM endpoint
- **Databases:** DRKG Neo4j container + local ADMET model weights
- **Hardware requirement:** single NVIDIA A100 (80GB) or dual A6000 (48GB × 2)

---

## 9. Multi-Modal Inputs: Vision, Structure & Omics

### 9.1 Histology + Spatial Transcriptomics

Clinical researchers increasingly work with histopathology images alongside genomics data. CellType-Agent should accept:

- `.tiff` / `.svs` histology slides → use **CONCH** or **UNI** (pathology vision-language models) to extract morphological embeddings → combine with transcriptomics for biomarker discovery
- Spatial transcriptomics (`.h5ad` with spatial coordinates) → use **Nicheformer** (2025) to model cell-cell communication in the tumor microenvironment
- **LLaVA-Med** or **BioViL-T** for image-caption generation from microscopy results — summarize IHC/IF images into structured biological claims

### 9.2 Structure File Inputs

Accept `.pdb` files directly in the CLI (`ct "Analyze this structure for druggable pockets" --file target.pdb`):

- **Pocket detection:** fpocket / AutoSite → identify druggable sites → feed into Boltz-2 for affinity screening
- **Structure comparison:** TM-align → detect allosteric site similarity to known drug targets
- **Hotspot analysis:** BindCraft / EvoEF2 → identify key interface residues for BoltzGen design constraints

### 9.3 `.h5ad` / Single-Cell Inputs

Accept Anndata (`.h5ad`) files for direct in-context analysis:

- Auto-detect if scRNA-seq or spatial; run appropriate foundation model
- Perform in-silico perturbation with compound SMILES: *"What happens to this tumor sample if I add compound X?"*
- Return differential gene expression as a structured tool result → feed into pathway enrichment

---

## 10. Reinforcement Learning from Experimental Feedback (RLEF)

The most transformative long-term enhancement is closing the loop between AI predictions and wet-lab outcomes. This transforms CellType-Agent from a static system into one that **improves with every experiment**.

### 10.1 Architecture

- After each research cycle, the researcher marks predictions as: `Validated` | `Partially Validated` | `Refuted`
- Each outcome is stored as a `(query, tool_calls, prediction, outcome)` tuple in a feedback database
- **Weekly:** run DPO training on local LoRA model using `validated = preferred`, `refuted = rejected`
- **Monthly:** update tool selection prior probabilities based on which tools contributed to validated predictions

### 10.2 Active Learning for Compound Screening

- **Bayesian optimization loop:** run Boltz-2 virtual screen → select 10 diverse top candidates → wet-lab IC50 → update surrogate model → next iteration
- Use **BOTORCH** (PyTorch Bayesian Optimization) as the acquisition function framework
- Each 10-compound batch costs ~$2,000–$5,000 in wet-lab work; AI reduces required iterations from 10+ to **3–4**
- **Persistence:** `ct` saves the active learning state between sessions — `ct resume --campaign CRBN_degraders`

---

## 11. Implementation Roadmap: 9-Month Plan

### Phase 1 (Months 1–2): Deep Generative Chemistry

> **Goal:** Make the agent a creator, not just a searcher. Every query about a target should end with a set of novel designed molecules.

- **Week 1–2:** Deepen Boltz-2 integration — expose `affinity_pred_value` for SAR, add pocket constraint interface
- **Week 3–4:** Integrate BoltzGen — `/tool add boltzgen` with target FASTA → returns top 10 de novo binder sequences + structures
- **Week 5–6:** Add ESM3 Forge API — `/tool add esm3-design` for conditional protein generation with off-target avoidance constraints
- **Week 7–8:** Add ADMET-AI + ChemProp — `/tool add admet-gnn` returning 41 endpoints per SMILES string

**Deliverables:** 4 new GPU-accelerated tools in `/tools`. Updated BixBench run targeting **92%+ accuracy**.

---

### Phase 2 (Months 3–5): Knowledge Graph & Memory

> **Goal:** Ground the agent in strict biological topology. Eliminate hallucinated pathway connections.

- **Month 3:** Deploy DRKG in Neo4j via `ct data pull drkg` — automatic graph construction from public data
- **Month 3:** Build GraphRAG layer — `/tool add biokg-query` with natural-language Cypher interface
- **Month 4:** Extend KG with local DepMap data — co-essentiality networks as graph edges
- **Month 4:** Add persistent vector memory (Qdrant) — agent remembers previous session findings, cites past experiments
- **Month 5:** Add scGPT + Arc Virtual Cell — `/tool add virtual-cell` for in-silico TME perturbation
- **Month 5:** Add Nicheformer for spatial transcriptomics analysis

**Deliverables:** `ct data pull drkg`, `ct memory` commands, 3 new biology tools, interactive KG explorer in HTML reports.

---

### Phase 3 (Months 6–8): Multi-Agent System

> **Goal:** Transform `/agents N` into a structured Multi-Agent System with specialized roles and adversarial critique.

- **Month 6:** Implement specialist agent prompts — `ct-chemist`, `ct-biologist`, `ct-toxicologist` (critic), `ct-orchestrator`
- **Month 6:** Add shared JSON workspace between sub-agents — structured finding exchange with confidence scores
- **Month 7:** Build the adversarial critique loop — `ct-toxicologist` reviews chemist's compounds and must find at least 3 failure modes before proceeding
- **Month 7:** Add DMTA-cycle commands — `/dmta-design`, `/dmta-make` (synthesizability check), `/dmta-test` (assay recommendation), `/dmta-analyze`
- **Month 8:** Implement Robin-style hypothesis–experiment–result loop for long-running campaigns

**Deliverables:** Structured multi-agent architecture, DMTA commands, adversarial critic, `/campaign` mode for multi-session research programs.

---

### Phase 4 (Month 9): Enterprise & Local LLM

> **Goal:** Enable fully offline, privacy-preserving deployment for pharma/biotech teams.

- **Month 9:** Release `celltype-agent-local` Docker container — quantized 70B model + vLLM + DRKG Neo4j + ADMET models
- **Month 9:** BixBench benchmark on local model stack — target **>80% accuracy**
- **Month 9:** LoRA training script — so enterprise customers can fine-tune on their proprietary experimental data
- **Month 9:** RLEF pipeline — automated DPO training from session feedback labels

**Deliverables:** Docker image on GitHub Packages, enterprise deployment guide, LoRA fine-tuning cookbook.

---

## 12. Key Open-Source Projects to Integrate

| Project | GitHub | License | Use Case |
|---|---|---|---|
| Boltz-2 + BoltzGen | `jwohlwend/boltz`, `HannesStark/boltzgen` | MIT | Structure + affinity + de novo design |
| ESM3 (1.4B) | `evolutionaryscale/esm` | CC-BY-NC 4.0 | Conditional protein generation |
| scGPT | `bowang-lab/scGPT` | MIT | Single-cell perturbation prediction |
| ADMET-AI | `swansonk14/admet_ai` | MIT | 41-endpoint ADMET prediction |
| ChemProp | `chemprop/chemprop` | MIT | Molecular property GNN |
| DRKG | `gnn4dr/DRKG` | CC-BY 4.0 | Drug repurposing knowledge graph |
| DeepPurpose | `kexinhuang12138/DeepPurpose` | BSD-3 | Drug-target interaction GNN |
| BOTORCH | `pytorch/botorch` | MIT | Bayesian optimization active learning |
| Qdrant | `qdrant/qdrant` | Apache 2.0 | Vector memory for session persistence |
| PyG (PyTorch Geometric) | `pyg-team/pytorch_geometric` | MIT | GNN backbone for polypharmacology |
| Robin | FutureHouse platform | Proprietary | Multi-agent scientific discovery pattern |
| GRNFormer | ACL 2025 paper | Research | Gene regulatory network from scRNA |

---

## 13. Projected Performance Targets

| Enhancement | Current State | Target (Post-Integration) | Timeline |
|---|---|---|---|
| BixBench Accuracy | 90.0% (Opus 4.6) | 93%+ with KG grounding | Phase 2 |
| Virtual Screening Speed | API-dependent | 10K compounds/hour (Boltz-2) | Phase 1 |
| Binding Affinity Accuracy | Qualitative (ChEMBL lookup) | 0.6 Pearson (FEP+ benchmark) | Phase 1 |
| Hallucination Rate (Pathways) | ~15% (flat RAG) | <3% (GraphRAG + DRKG) | Phase 2 |
| De Novo Design Capability | None | 66% nanomolar hit rate (BoltzGen) | Phase 1 |
| Off-Target Liability Coverage | Anti-target flags only | Full proteome GNN screening | Phase 2 |
| Session Memory Depth | Single session | Indefinite (Qdrant vector DB) | Phase 2 |
| Wet-Lab Iteration Cycles | Not applicable | 3–4 cycles (active learning) | Phase 3 |

---

## 14. Conclusion: The In-Silico Scientist Vision

CellType-Agent has already established itself as the best-in-class autonomous agent for computational drug discovery. The roadmap outlined here is not incremental — it is a complete **paradigm shift** from tool-calling to generative science.

The convergence of five simultaneous breakthroughs in 2025 makes this the ideal moment to execute this vision:

1. **Boltz-2 + BoltzGen** (MIT/Recursion): binding affinity prediction + de novo binder design at 1000× FEP speed, fully open-source
2. **ESM3** (EvolutionaryScale): controllable protein generation with function-level constraints, published in *Science*
3. **Arc Virtual Cell**: 50% better perturbation prediction than prior generation models, trained on Tahoe-100M dataset
4. **Robin** (FutureHouse): proof that a 3-agent system can conduct and validate real drug discovery in 2.5 months
5. **GraphRAG in Pharma**: Merck, Bayer, and Lilly all deploying graph-grounded AI — the infrastructure is proven

By Phase 3 completion, CellType-Agent will be capable of receiving a natural language query like:

> *"I have a novel CRBN glue that degrades GSPT1. Design a backup molecule that avoids SALL4 and CK1alpha off-targets, predict its binding affinity, model how a triple-negative breast cancer TME responds, identify the top resistance biomarkers, and propose a rational combination to overcome them."*

...and autonomously executing a complete multi-step research program — designing molecules with BoltzGen, validating affinity with Boltz-2, running in-silico perturbation with the Arc Virtual Cell, traversing the bioKG for resistance biomarkers, and delivering a comprehensive HTML research report — all without a single manual step.

**That is the In-Silico Scientist. That is the next version of CellType-Agent.**

---

*CellType-Agent Enhancement Report | March 2026 | celltype.com*
*github.com/celltype/celltype-agent | MIT Licensed | Built by CellType Inc.*
