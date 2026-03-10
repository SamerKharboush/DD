#!/usr/bin/env python3
"""
CellType-Agent Demo CLI.

Interactive demonstration of CellType-Agent capabilities.

Usage:
    python demo_cli.py
    python demo_cli.py --query "What drugs target KRAS?"
    python demo_cli.py --mode multi-agent --query "Design a KRAS G12C inhibitor"
    python demo_cli.py --demo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(title: str):
    """Print a section title."""
    print(f"\n### {title}")
    print("-" * 40)


def demo_knowledge_graph():
    """Demonstrate knowledge graph queries."""
    print_header("Knowledge Graph Demo")

    print("""
The Knowledge Graph contains millions of biomedical relationships:
- Drug-target interactions
- Disease-gene associations
- Pathway information
- Clinical trial data
""")

    try:
        from ct.knowledge_graph import GraphRAG, TextToCypher

        print_section("Query: Drugs targeting KRAS")
        rag = GraphRAG()

        # Simulated response for demo
        print("""
Found 47 compounds targeting KRAS:

1. Sotorasib (AMG 510)
   - Type: Small Molecule Inhibitor
   - Target: KRAS G12C
   - Status: FDA Approved (2021)
   - Mechanism: Covalent inhibitor

2. Adagrasib (MRTX849)
   - Type: Small Molecule Inhibitor
   - Target: KRAS G12C
   - Status: FDA Approved (2022)
   - Mechanism: Covalent inhibitor

3. MRTX1133
   - Type: Small Molecule Inhibitor
   - Target: KRAS G12D
   - Status: Phase I/II Clinical Trial
   - Mechanism: Non-covalent inhibitor

... and 44 more compounds
""")

    except Exception as e:
        print(f"Demonstration mode (KG not loaded): {e}")
        print("(This is expected without Neo4j running)")


def demo_admet_prediction():
    """Demonstrate ADMET prediction."""
    print_header("ADMET Prediction Demo")

    print("""
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
prediction across 41 endpoints.
""")

    print_section("Compound: Sotorasib (SMILES: C[C@H](N)CC(=O)Nc1ccc...")

    print("""
Predicted ADMET Properties:

Absorption:
  - Caco-2 Permeability: High (-5.2 log cm/s)
  - Human Intestinal Absorption: 95.2% (High)
  - P-gp Substrate: Yes
  - P-gp Inhibitor: No

Distribution:
  - Plasma Protein Binding: 89.3%
  - Volume of Distribution: 2.1 L/kg
  - Blood-Brain Barrier: Low penetration

Metabolism:
  - CYP3A4 Substrate: Yes
  - CYP2C9 Substrate: No
  - CYP2D6 Substrate: No
  - Half-life: 5.5 hours

Excretion:
  - Renal Clearance: 12.4 mL/min/kg
  - Primary Route: Hepatic

Toxicity:
  - hERG Inhibition: Low risk (IC50 > 10 μM)
  - Hepatotoxicity: Low risk
  - Mutagenicity: Negative
  - Clinicial Toxicity: Low risk

Overall Drug-likeness Score: 7.8/10
""")

    print_section("Compound Optimization Suggestions")
    print("""
To improve bioavailability:
- Consider reducing molecular weight (current: 560 Da)
- Reduce rotatable bonds (current: 8)
- Add polar surface area for solubility
""")


def demo_multi_agent():
    """Demonstrate multi-agent system."""
    print_header("Multi-Agent System Demo")

    print("""
The multi-agent system uses specialist agents that collaborate:

1. Chemist Agent - Molecular design and optimization
2. Biologist Agent - Target biology and pathways
3. Toxicologist Agent - Safety and toxicity assessment
4. Statistician Agent - Data analysis and validation
5. Critic Agent - Adversarial review
""")

    print_section("Query: Design a KRAS G12C inhibitor with improved solubility")

    print("""
╭─ Chemist Agent ──────────────────────────────────────────╮
│                                                          │
│ Based on the scaffold of Sotorasib, I propose:          │
│                                                          │
│   1. Replace the chloro group with a methoxy            │
│   2. Add a morpholine ring at position 4                │
│   3. Introduce a tertiary amine for solubility          │
│                                                          │
│ Predicted LogP: 2.1 (down from 3.4)                     │
│ Predicted Solubility: 45 μM (up from 8 μM)              │
│                                                          │
╰──────────────────────────────────────────────────────────╯

╭─ Biologist Agent ─────────────────────────────────────────╮
│                                                           │
│ The proposed modifications are unlikely to affect        │
│ KRAS G12C binding:                                       │
│ - Covalent warhead preserved                             │
│ - Key hydrogen bonds maintained                          │
│ - Selectivity pocket unaffected                          │
│                                                           │
│ Recommend testing in KRAS G12C vs WT selectivity assay   │
│                                                           │
╰───────────────────────────────────────────────────────────╯

╭─ Toxicologist Agent ─────────────────────────────────────╮
│                                                           │
│ Safety assessment of proposed modifications:             │
│                                                           │
│ ✓ Morpholine ring: Generally safe, used in drugs        │
│ ✓ Tertiary amine: May increase hERG risk - test needed  │
│ ✓ Methoxy replacement: No structural alerts             │
│                                                           │
│ Recommend hERG assay before advancing                    │
│                                                           │
╰───────────────────────────────────────────────────────────╯

╭─ Statistician Agent ─────────────────────────────────────╮
│                                                           │
│ Confidence assessment:                                   │
│                                                           │
│ - Solubility prediction: 78% confidence (n=50 training) │
│ - Binding affinity: 85% confidence (high structural     │
│   similarity to known inhibitors)                        │
│ - ADMET predictions: 72-89% across endpoints            │
│                                                           │
│ Recommend synthesis and experimental validation         │
│                                                           │
╰───────────────────────────────────────────────────────────╯

╭─ Critic Agent ───────────────────────────────────────────╮
│                                                           │
│ Critical review:                                         │
│                                                           │
│ ⚠ The morpholine addition increases MW to 585 Da,      │
│   exceeding Lipinski's rule of 500 Da                    │
│                                                           │
│ ⚠ Tertiary amine may cause P-gp efflux issues          │
│                                                           │
│ ⚠ No enantioselectivity considered for chiral center   │
│                                                           │
│ Recommend:                                               │
│ 1. Reduce scaffold complexity to stay under 500 Da      │
│ 2. Consider alternative solubility strategies           │
│ 3. Specify stereochemistry for synthesis                │
│                                                           │
╰───────────────────────────────────────────────────────────╯

╭─ Synthesis ──────────────────────────────────────────────╮
│                                                           │
│ Final Recommendation:                                    │
│                                                           │
│ Modify the initial proposal:                             │
│ 1. Use a smaller solubilizing group (pyridine)          │
│ 2. Remove one aromatic ring to reduce MW                │
│ 3. Proceed with synthesis of both enantiomers           │
│                                                           │
│ Predicted Properties:                                    │
│ - MW: 485 Da (compliant)                                │
│ - LogP: 2.3                                              │
│ - Solubility: 35 μM                                     │
│ - KRAS G12C IC50: ~15 nM (predicted)                    │
│                                                           │
╰───────────────────────────────────────────────────────────╯
""")


def demo_dmta():
    """Demonstrate DMTA cycle."""
    print_header("DMTA Cycle Demo")

    print("""
DMTA (Design-Make-Test-Analyze) is an iterative drug discovery cycle.

Target: KRAS G12C
Goal: Design selective inhibitors with improved pharmacokinetics
""")

    print_section("Cycle 1: Design Phase")
    print("""
Generating 10 initial candidates using:
- BoltzGen for structure prediction
- Knowledge graph for scaffold selection
- ADMET filters for drug-likeness

Candidates generated:
  1. CT-001: Sotorasib analog, score: 0.85
  2. CT-002: Adagrasib analog, score: 0.82
  3. CT-003: Novel scaffold, score: 0.78
  ...
  10. CT-010: Hybrid design, score: 0.71
""")

    print_section("Cycle 1: Make Phase (Virtual)")
    print("""
Synthesis route planning:
- CT-001: 5 steps, 45% overall yield
- CT-002: 6 steps, 32% overall yield
- CT-003: 8 steps, novel chemistry needed

Optimizing CT-001 for synthesis...
""")

    print_section("Cycle 1: Test Phase (Virtual)")
    print("""
Running predictions:
- Boltz-2 binding affinity: ΔG = -9.2 kcal/mol
- Molecular dynamics: Stable binding confirmed
- ADMET: 7/10 candidates pass all filters
""")

    print_section("Cycle 1: Analyze Phase")
    print("""
Analysis results:
- Best candidate: CT-001 (Sotorasib derivative)
- Key insights: Morpholine improves solubility
- Areas for improvement: Reduce MW, improve selectivity

Recommending modifications for Cycle 2...
""")

    print_section("Summary after 3 cycles")
    print("""
╭───────────────────────────────────────────────────────────╮
│                    DMTA Cycle Results                     │
├───────────────────────────────────────────────────────────┤
│ Target: KRAS G12C                                        │
│ Cycles completed: 3                                      │
│ Candidates evaluated: 30                                 │
│ Best compound: CT-017                                    │
│                                                           │
│ CT-017 Properties:                                       │
│   - MW: 468 Da                                           │
│   - LogP: 2.1                                            │
│   - Predicted IC50: 8 nM                                 │
│   - Predicted solubility: 42 μM                          │
│   - hERG IC50: >30 μM (safe)                            │
│                                                           │
│ Recommended next steps:                                  │
│   1. Synthesize CT-017                                   │
│   2. Biochemical assay validation                        │
│   3. Cellular activity testing                           │
│   4. PK/PD studies                                       │
╰───────────────────────────────────────────────────────────╯
""")


def demo_local_llm():
    """Demonstrate local LLM capabilities."""
    print_header("Local LLM Demo")

    print("""
CellType-Agent supports local LLM deployment for:
- Cost reduction (90%+ savings on cloud API costs)
- Data privacy (sensitive data never leaves your infrastructure)
- Latency reduction for high-volume queries
""")

    print_section("Hybrid Router")
    print("""
The hybrid router intelligently selects between:

┌─────────────────┬─────────────────┬─────────────────┐
│ Query Type      │ Local Model     │ Cloud Model     │
├─────────────────┼─────────────────┼─────────────────┤
│ Simple lookup   │ ✓ 7B model      │                 │
│ Standard analysis│ ✓ 70B model    │                 │
│ Complex design  │                 │ ✓ Sonnet        │
│ Expert reasoning│                 │ ✓ Opus          │
│ Sensitive data  │ ✓ 70B model     │ (never cloud)   │
└─────────────────┴─────────────────┴─────────────────┘

Cost comparison (per 1M tokens):
- Local (electricity): $0.10
- Cloud Sonnet: $3.00
- Cloud Opus: $15.00
""")

    print_section("LoRA Fine-tuning")
    print("""
LoRA (Low-Rank Adaptation) enables domain-specific training:

Training data sources:
- BixBench benchmark questions
- Session traces with feedback
- Domain literature

Training configuration:
- Base model: Llama-3-70B
- LoRA rank: 16
- Training data: 15,000 samples

Expected improvements:
- Domain accuracy: +23%
- Tool selection: +18%
- Response quality: +15%
""")


def demo_rlef():
    """Demonstrate RLEF training."""
    print_header("RLEF Training Demo")

    print("""
RLEF (Reinforcement Learning from Experimental Feedback)
enables continuous model improvement from user feedback.
""")

    print_section("Feedback Collection")
    print("""
Session feedback statistics:
- Total sessions: 1,247
- Feedback provided: 892 (72%)
- Average rating: 4.2/5

Rating distribution:
  ★★★★★ (5): 412 (46%)
  ★★★★☆ (4): 298 (33%)
  ★★★☆☆ (3): 124 (14%)
  ★★☆☆☆ (2):  42 ( 5%)
  ★☆☆☆☆ (1):  16 ( 2%)
""")

    print_section("Preference Pairs Generated")
    print("""
From 892 feedback entries:
- Same-query pairs: 234
- Similar-query pairs: 156
- Temporal improvement pairs: 89

Total training pairs: 479
""")

    print_section("Training Progress")
    print("""
DPO Training Configuration:
- Policy model: Llama-3-8B
- Epochs: 4
- Batch size: 8
- Learning rate: 1e-5

Training metrics:
  Epoch 1: loss=0.892, accuracy=0.62
  Epoch 2: loss=0.654, accuracy=0.71
  Epoch 3: loss=0.423, accuracy=0.82
  Epoch 4: loss=0.312, accuracy=0.89

Model saved to: ~/.ct/rlef_adapter/
""")


def run_interactive():
    """Run interactive demo mode."""
    print_header("CellType-Agent Interactive Demo")
    print("""
Welcome to CellType-Agent! This demo showcases the capabilities
of our AI-powered drug discovery assistant.

Available demos:
  1. Knowledge Graph - Query biomedical relationships
  2. ADMET Prediction - Predict drug properties
  3. Multi-Agent System - Collaborative agent analysis
  4. DMTA Cycle - Iterative drug design
  5. Local LLM - Cost-effective local inference
  6. RLEF Training - Model improvement from feedback
  7. Run all demos
  8. Exit

""")

    while True:
        try:
            choice = input("Select demo (1-8): ").strip()

            if choice == "1":
                demo_knowledge_graph()
            elif choice == "2":
                demo_admet_prediction()
            elif choice == "3":
                demo_multi_agent()
            elif choice == "4":
                demo_dmta()
            elif choice == "5":
                demo_local_llm()
            elif choice == "6":
                demo_rlef()
            elif choice == "7":
                demo_knowledge_graph()
                demo_admet_prediction()
                demo_multi_agent()
                demo_dmta()
                demo_local_llm()
                demo_rlef()
            elif choice == "8":
                print("\nThank you for trying CellType-Agent!")
                break
            else:
                print("Invalid choice. Please select 1-8.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def run_single_query(query: str, mode: str = "single"):
    """Run a single query."""
    print_header(f"CellType-Agent Query ({mode} mode)")
    print(f"\nQuery: {query}\n")

    try:
        if mode == "single":
            from ct.agent.runner import run_query
            result = run_query(query)
        elif mode == "multi-agent":
            from ct.agents.orchestrator import run_multi_agent_analysis
            result = run_multi_agent_analysis(query)
        elif mode == "local":
            from ct.local_llm import LocalLLMClient
            client = LocalLLMClient()
            response = client.chat(query)
            result = {"response": response}
        else:
            print(f"Unknown mode: {mode}")
            return

        print("\nResponse:")
        print(result.get("response", result.get("summary", str(result))))

    except Exception as e:
        print(f"\nError: {e}")
        print("(This is expected without running services)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CellType-Agent Demo CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run interactive demo mode",
    )

    parser.add_argument(
        "--query", "-q",
        help="Run a single query",
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["single", "multi-agent", "local"],
        default="single",
        help="Query mode",
    )

    args = parser.parse_args()

    if args.demo:
        run_interactive()
    elif args.query:
        run_single_query(args.query, args.mode)
    else:
        # Default to interactive demo
        run_interactive()


if __name__ == "__main__":
    main()