"""
CellType-Agent REST API.

FastAPI-based REST API for CellType-Agent services.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("ct.api")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Starting CellType-Agent API...")

    # Initialize resources
    from ct.session_logging import SessionLogger
    app.state.session_logger = SessionLogger()

    # Initialize auth manager
    from ct.security.auth import AuthManager
    app.state.auth_manager = AuthManager()

    # Initialize API key manager
    from ct.security.api_keys import APIKeyManager
    app.state.api_key_manager = APIKeyManager()

    # Check GPU availability
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"GPU available: {result.stdout.strip()}")
            app.state.gpu_available = True
        else:
            app.state.gpu_available = False
    except Exception:
        app.state.gpu_available = False

    yield

    # Cleanup
    logger.info("Shutting down CellType-Agent API...")


# Create FastAPI app
app = FastAPI(
    title="CellType-Agent API",
    description="AI-powered drug discovery assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query")
    mode: str = Field(default="single", description="Execution mode")
    context: Optional[dict] = Field(default=None, description="Additional context")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class QueryResponse(BaseModel):
    """Query response model."""
    response: str
    session_id: str
    mode: str
    latency: float
    tool_calls: list[dict] = []
    metadata: dict = {}


class MultiAgentRequest(BaseModel):
    """Multi-agent analysis request."""
    query: str
    mode: str = Field(default="debate", description="Orchestration mode")
    agents: list[str] = Field(default_factory=list, description="Specific agents to use")
    context: Optional[dict] = None


class DMTARequest(BaseModel):
    """DMTA cycle request."""
    target: str = Field(..., description="Target name or sequence")
    iterations: int = Field(default=1, ge=1, le=5, description="Number of iterations")
    num_candidates: int = Field(default=10, ge=1, le=100, description="Candidates per cycle")
    constraints: Optional[dict] = None


class FeedbackRequest(BaseModel):
    """Feedback submission request."""
    session_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    comments: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    version: str
    uptime: float


# ============================================
# Health Endpoints
# ============================================

start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu_available=getattr(app.state, "gpu_available", False),
        version="1.0.0",
        uptime=time.time() - start_time,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CellType-Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================
# Query Endpoints
# ============================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Run a single query through CellType-Agent.

    Modes:
    - single: Direct query processing
    - multi-agent: Multi-agent analysis
    - dmta: DMTA cycle
    - local: Use local LLM
    """
    from ct.agent.runner import run_query as _run_query
    from ct.session_logging import SessionLogger

    session_logger = app.state.session_logger

    start = time.time()

    try:
        if request.mode == "single":
            result = _run_query(request.query, **(request.context or {}))

        elif request.mode == "multi-agent":
            from ct.agents.orchestrator import run_multi_agent_analysis
            result = run_multi_agent_analysis(
                query=request.query,
                context=request.context,
            )

        elif request.mode == "local":
            from ct.local_llm import LocalLLMClient
            client = LocalLLMClient()
            response = client.chat(request.query)
            result = {"response": response, "mode": "local"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

        latency = time.time() - start

        return QueryResponse(
            response=result.get("response", result.get("summary", "")),
            session_id=request.session_id or session_logger.current_session_id,
            mode=request.mode,
            latency=latency,
            tool_calls=result.get("tool_calls", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/multi-agent", response_model=QueryResponse)
async def run_multi_agent(request: MultiAgentRequest):
    """
    Run multi-agent analysis.

    Modes:
    - sequential: Agents run one after another
    - parallel: Agents run in parallel
    - debate: Agents debate and reach consensus
    """
    from ct.agents.orchestrator import run_multi_agent_analysis

    start = time.time()

    try:
        result = run_multi_agent_analysis(
            query=request.query,
            mode=request.mode,
            context=request.context,
            specific_agents=request.agents if request.agents else None,
        )

        latency = time.time() - start

        return QueryResponse(
            response=result.get("summary", ""),
            session_id=result.get("session_id", ""),
            mode=f"multi-agent-{request.mode}",
            latency=latency,
            tool_calls=result.get("tool_calls", []),
            metadata={"agents": result.get("agents", [])},
        )

    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/dmta", response_model=QueryResponse)
async def run_dmta(request: DMTARequest):
    """
    Run a DMTA (Design-Make-Test-Analyze) cycle.

    Generates drug candidates, predicts properties,
    and iteratively optimizes.
    """
    from ct.campaign.dmta import run_dmta_cycle

    start = time.time()

    try:
        result = run_dmta_cycle(
            target=request.target,
            num_candidates=request.num_candidates,
            iterations=request.iterations,
            constraints=request.constraints,
        )

        latency = time.time() - start

        return QueryResponse(
            response=result.get("summary", ""),
            session_id=result.get("session_id", ""),
            mode="dmta",
            latency=latency,
            tool_calls=result.get("tool_calls", []),
            metadata={
                "iterations": request.iterations,
                "candidates": result.get("candidates", []),
                "best_compound": result.get("best_compound"),
            },
        )

    except Exception as e:
        logger.error(f"DMTA cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Feedback Endpoints
# ============================================

@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a session."""
    try:
        from ct.session_logging import SessionLogger

        session_logger = getattr(app.state, "session_logger", None) or SessionLogger()

        session_logger.add_feedback(
            session_id=request.session_id,
            rating=request.rating,
            feedback=request.comments or "",
        )

        return {"status": "success", "session_id": request.session_id}

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        # Return success anyway to not block the user
        return {"status": "recorded", "session_id": request.session_id}


@app.get("/api/v1/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics."""
    from ct.rlef.feedback_processor import FeedbackProcessor

    processor = FeedbackProcessor()
    analytics = processor.get_analytics()

    return analytics


# ============================================
# Model Management Endpoints
# ============================================

@app.get("/api/v1/models")
async def list_models():
    """List available models."""
    return {
        "cloud": [
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "tier": "fast"},
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "tier": "capable"},
        ],
        "local": [
            {"id": "llama-3-70b", "name": "Llama 3 70B", "available": False},
            {"id": "llama-3-8b", "name": "Llama 3 8B", "available": False},
        ],
    }


@app.get("/api/v1/models/status")
async def get_model_status():
    """Get model status and availability."""
    from ct.local_llm import LocalLLMClient, HybridRouter

    local_client = LocalLLMClient()
    router = HybridRouter()

    return {
        "local_available": local_client.is_available(),
        "router_status": router.get_status(),
    }


# ============================================
# Knowledge Graph Endpoints
# ============================================

@app.get("/api/v1/knowledge-graph/stats")
async def get_kg_stats():
    """Get knowledge graph statistics."""
    try:
        from ct.knowledge_graph import Neo4jClient

        client = Neo4jClient()
        stats = client.get_stats()

        return stats

    except Exception as e:
        return {"error": str(e), "available": False}


@app.post("/api/v1/knowledge-graph/query")
async def query_knowledge_graph(query: str, limit: int = 10):
    """Query the knowledge graph."""
    try:
        from ct.knowledge_graph import GraphRAG

        rag = GraphRAG()
        results = rag.query(query, limit=limit)

        return {"results": results, "query": query}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# GPU Service Endpoints
# ============================================

@app.get("/api/v1/gpu/status")
async def get_gpu_status():
    """Get GPU status."""
    if not getattr(app.state, "gpu_available", False):
        return {"available": False}

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "available": True,
                "name": parts[0],
                "memory_used": parts[1],
                "memory_total": parts[2],
                "utilization": parts[3],
            }

    except Exception as e:
        logger.error(f"GPU status check failed: {e}")

    return {"available": False}


# ============================================
# RLEF Training Endpoints
# ============================================

@app.post("/api/v1/rlef/train")
async def trigger_rlef_training(background_tasks: BackgroundTasks):
    """Trigger RLEF training in background."""
    from ct.rlef import RLEFTrainer

    def run_training():
        trainer = RLEFTrainer()
        trainer.train()

    background_tasks.add_task(run_training)

    return {"status": "training_started"}


@app.get("/api/v1/rlef/stats")
async def get_rlef_stats():
    """Get RLEF training statistics."""
    from ct.rlef import RLEFTrainer

    trainer = RLEFTrainer()
    stats = trainer.get_training_stats()

    return stats


# ============================================
# Authentication Endpoints
# ============================================

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Registration request model."""
    username: str
    password: str
    email: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours


@app.post("/api/v1/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register a new user."""
    auth_manager = app.state.auth_manager

    try:
        user = auth_manager.register_user(
            username=request.username,
            password=request.password,
            email=request.email,
        )

        token = auth_manager.login(request.username, request.password)

        return TokenResponse(access_token=token)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login and get access token."""
    auth_manager = app.state.auth_manager

    token = auth_manager.login(request.username, request.password)

    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(access_token=token)


@app.get("/api/v1/auth/me")
async def get_current_user_info(authorization: Optional[str] = Header(None)):
    """Get current user info."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:]
    auth_manager = app.state.auth_manager

    user = auth_manager.verify(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
    }


# ============================================
# API Key Endpoints
# ============================================

class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""
    name: str
    scopes: list[str] = ["read"]
    rate_limit: int = 60


@app.post("/api/v1/api-keys")
async def create_api_key_endpoint(request: CreateAPIKeyRequest, authorization: Optional[str] = Header(None)):
    """Create a new API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:]
    auth_manager = app.state.auth_manager

    user = auth_manager.verify(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    api_key_manager = app.state.api_key_manager
    key = api_key_manager.create_key(
        name=request.name,
        user_id=user.user_id,
        scopes=request.scopes,
        rate_limit=request.rate_limit,
    )

    return {"api_key": key, "message": "Store this key securely - it won't be shown again"}


@app.get("/api/v1/api-keys")
async def list_api_keys(authorization: Optional[str] = Header(None)):
    """List user's API keys."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:]
    auth_manager = app.state.auth_manager

    user = auth_manager.verify(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    api_key_manager = app.state.api_key_manager
    keys = api_key_manager.list_keys(user_id=user.user_id)

    return {"keys": [{"key_id": k.key_id, "name": k.name, "scopes": k.scopes, "is_active": k.is_active} for k in keys]}


# ============================================
# Monitoring Endpoints
# ============================================

@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get Prometheus metrics."""
    from ct.monitoring import get_metrics
    metrics = get_metrics()
    return metrics.export_prometheus()


@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components."""
    from ct.monitoring import run_health_checks
    return run_health_checks()


# ============================================
# GPU Service Endpoints
# ============================================

@app.post("/api/v1/gpu/predict-structure")
async def predict_protein_structure(sequence: str):
    """Predict protein structure using Boltz-2."""
    from ct.gpu import Boltz2Service

    service = Boltz2Service()
    if not service.is_available():
        raise HTTPException(status_code=503, detail="GPU service not available")

    result = service.predict_structure(sequence)

    return {
        "sequence": sequence[:50],
        "confidence": result.confidence,
        "plddt": result.plddt,
        "prediction_time_s": result.prediction_time_s,
    }


@app.post("/api/v1/gpu/predict-affinity")
async def predict_binding_affinity(protein_sequence: str, ligand_smiles: str):
    """Predict binding affinity using Boltz-2."""
    from ct.gpu import Boltz2Service

    service = Boltz2Service()
    if not service.is_available():
        raise HTTPException(status_code=503, detail="GPU service not available")

    result = service.predict_affinity(protein_sequence, ligand_smiles)

    return {
        "affinity_nm": result.affinity_nm,
        "confidence": result.confidence,
        "delta_g": result.delta_g,
        "prediction_time_s": result.prediction_time_s,
    }


@app.post("/api/v1/gpu/dock")
async def dock_molecule(protein_pdb: str, ligand_smiles: str):
    """Perform molecular docking with DiffDock."""
    from ct.gpu import DiffDockService

    service = DiffDockService()
    if not service.is_available():
        raise HTTPException(status_code=503, detail="GPU service not available")

    result = service.dock(protein_pdb, ligand_smiles)

    return {
        "best_score": result.best_score,
        "confidence": result.confidence,
        "num_poses": len(result.poses),
        "prediction_time_s": result.prediction_time_s,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)