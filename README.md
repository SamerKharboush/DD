# CellType-Agent

AI-powered drug discovery assistant with multi-agent orchestration, knowledge graph integration, and local LLM support.

## Overview

CellType-Agent is a comprehensive AI system for drug discovery research, featuring:

- **Knowledge Graph**: Integration with DRKG (Drug Repurposing Knowledge Graph) via Neo4j
- **ADMET Prediction**: 41-endpoint prediction using GNN models
- **Multi-Agent System**: Collaborative specialist agents (Chemist, Biologist, Toxicologist, Statistician, Critic)
- **DMTA Cycle**: Design-Make-Test-Analyze workflow for iterative drug design
- **Generative Chemistry**: Boltz-2 for structure prediction, ESM3 for protein design
- **Local LLM**: Cost-effective local inference with hybrid routing
- **RLEF Training**: Reinforcement Learning from Experimental Feedback

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/celltype-agent.git
cd celltype-agent

# Set environment variables
export ANTHROPIC_API_KEY=your-key-here
export NEO4J_PASSWORD=your-password

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ct-api
```

### Using pip

```bash
# Install dependencies
pip install -e .

# Run the CLI
ct "What drugs target KRAS?"

# Run interactive mode
ct --interactive

# Run multi-agent analysis
ct --mode multi-agent "Design a KRAS G12C inhibitor"

# Run DMTA cycle
ct --dmta --target "KRAS_G12C" --iterations 3
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CellType-Agent                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   CLI/API   │  │  Session    │  │  Feedback   │        │
│  │   Layer     │  │  Logging    │  │  Collector  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│  ┌──────┴────────────────┴────────────────┴──────┐        │
│  │              Hybrid Router                     │        │
│  │    (Local 7B/70B ↔ Cloud Sonnet/Opus)        │        │
│  └──────────────────────┬────────────────────────┘        │
│                         │                                  │
│  ┌──────────────────────┴────────────────────────┐        │
│  │            Multi-Agent Orchestrator            │        │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │        │
│  │  │Chemist │ │Biologist│ │Toxico- │ │Statis- │  │        │
│  │  │        │ │        │ │logist  │ │tician  │  │        │
│  │  └────────┘ └────────┘ └────────┘ └────────┘  │        │
│  │                    ┌────────┐                  │        │
│  │                    │ Critic │                  │        │
│  │                    └────────┘                  │        │
│  └──────────────────────┬────────────────────────┘        │
│                         │                                  │
│  ┌──────────────────────┴────────────────────────┐        │
│  │                  Tools Layer                   │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │        │
│  │  │Knowledge │ │  ADMET   │ │ Generative│       │        │
│  │  │  Graph   │ │Predictor │ │  Models   │       │        │
│  │  └──────────┘ └──────────┘ └──────────┘       │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │        │
│  │  │  Boltz-2 │ │ DiffDock │ │   ESM3   │       │        │
│  │  └──────────┘ └──────────┘ └──────────┘       │        │
│  └───────────────────────────────────────────────┘        │
│                                                             │
│  ┌───────────────────────────────────────────────┐        │
│  │              Infrastructure Layer              │        │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │        │
│  │  │ Neo4j  │ │ Redis  │ │Postgres│ │ Qdrant │  │        │
│  │  └────────┘ └────────┘ └────────┘ └────────┘  │        │
│  │  ┌────────┐ ┌────────┐ ┌────────┐             │        │
│  │  │ vLLM   │ │  GPU   │ │Prom/   │             │        │
│  │  │ Server │ │  Mgr   │ │Grafana │             │        │
│  │  └────────┘ └────────┘ └────────┘             │        │
│  └───────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Phase 1: Foundation (Months 1-6)
- ✅ Knowledge Graph (DRKG, Neo4j, GraphRAG)
- ✅ ADMET Prediction (41 endpoints)
- ✅ GPU Infrastructure (resource management, batch processing)
- ✅ Session Logging (traces, feedback collection)

### Phase 2: Generative Chemistry (Months 7-9)
- ✅ Boltz-2 Integration (structure prediction)
- ✅ BoltzGen (de novo design)
- ✅ ESM3 Client (protein generation)
- ✅ Validation Pipeline
- ✅ Structure I/O (PDB, H5AD, FASTA)

### Phase 3: Multi-Agent System (Months 10-12)
- ✅ Specialist Agents (Chemist, Biologist, Toxicologist, Statistician)
- ✅ Critic Agent (adversarial review)
- ✅ Multi-Agent Orchestrator
- ✅ Vector Memory (semantic search)
- ✅ DMTA Cycle

### Phase 4: Multi-Agent Expansion (Months 13-15)
- ✅ Enhanced orchestration modes
- ✅ Agent communication protocols
- ✅ Parallel execution
- ✅ Consensus mechanisms

### Phase 5: Local LLM + RLEF (Months 16-18)
- ✅ vLLM Server Integration
- ✅ LoRA Fine-tuning
- ✅ Hybrid Router
- ✅ RLEF Training (DPO, KTO)
- ✅ Feedback Processing

## API Reference

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Run a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What drugs target KRAS?"}'

# Multi-agent analysis
curl -X POST http://localhost:8000/api/v1/multi-agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Design a KRAS G12C inhibitor", "mode": "debate"}'

# DMTA cycle
curl -X POST http://localhost:8000/api/v1/dmta \
  -H "Content-Type: application/json" \
  -d '{"target": "KRAS_G12C", "iterations": 3}'

# Submit feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "rating": 5}'
```

### CLI Commands

```bash
# Single query
ct "What drugs target KRAS?"

# Multi-agent mode
ct --mode multi-agent "Analyze this compound"

# DMTA cycle
ct --dmta --target "KRAS_G12C" --iterations 3

# Local LLM
ct --local --model llama-3-70b "Query with local model"

# Interactive mode
ct --interactive

# RLEF training
ct --rlef-train sessions.jsonl
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | Required |
| `ESM3_API_KEY` | ESM3 API key | Optional |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `DATABASE_URL` | PostgreSQL URL | `postgresql://...` |
| `GPU_ENABLED` | Enable GPU services | `false` |

### Configuration Files

- `config/settings.yaml` - Main configuration
- `config/agents.yaml` - Agent definitions
- `config/tools.yaml` - Tool configurations

## Development

### Setup Development Environment

```bash
# Clone and install dev dependencies
git clone https://github.com/your-org/celltype-agent.git
cd celltype-agent
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/
mypy src/

# Format code
black src/
```

### Project Structure

```
celltype-agent/
├── src/ct/
│   ├── __main__.py          # CLI entry point
│   ├── api/                  # REST API
│   ├── agent/                # Core agent logic
│   ├── agents/               # Multi-agent system
│   ├── knowledge_graph/      # KG integration
│   ├── admet/                # ADMET prediction
│   ├── generative/           # Generative models
│   ├── local_llm/            # Local LLM support
│   ├── rlef/                 # RLEF training
│   ├── campaign/             # DMTA cycle
│   ├── memory/               # Vector memory
│   ├── session_logging/      # Session management
│   └── tools/                # Tool implementations
├── tests/                    # Test suite
├── deploy/                   # Deployment configs
├── data/                     # Data files
├── models/                   # Model weights
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start with GPU support
docker-compose --profile gpu up -d

# Start with local LLM
docker-compose --profile local-llm up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Scale API instances
docker-compose up -d --scale ct-api=3
```

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/

# Check deployment status
kubectl get pods -l app=celltype-agent
```

## Monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Neo4j Browser: http://localhost:7474

## License

MIT License - see LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.

## Citation

If you use CellType-Agent in your research, please cite:

```bibtex
@software{celltype-agent,
  title = {CellType-Agent: AI-Powered Drug Discovery Assistant},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/your-org/celltype-agent}
}
```