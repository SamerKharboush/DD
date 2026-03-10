"""
CellType-Agent Main Entry Point.

Integrates all phases:
- Phase 1: Knowledge Graph, ADMET, GPU Infrastructure
- Phase 2: Generative Chemistry (BoltzGen, ESM3)
- Phase 3: Multi-Agent System
- Phase 4: Advanced Multi-Agent + DMTA
- Phase 5: Local LLM + RLEF

Usage:
    ct "Design a KRAS G12C inhibitor"
    ct --mode multi-agent "Analyze this compound"
    ct --dmta --target "KRAS_G12C"
    ct --local "Query with local model"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ct")

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_environment():
    """Setup environment and configuration."""
    # Check for API keys
    api_keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "ESM3_API_KEY": os.environ.get("ESM3_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }

    missing = [k for k, v in api_keys.items() if not v]
    if missing:
        logger.warning(f"Missing API keys: {', '.join(missing)}")

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"GPU available: {result.stdout.strip()}")
    except Exception:
        logger.info("No GPU detected - will use CPU fallback")

    return api_keys


def run_single_query(query: str, context: Optional[dict] = None) -> dict:
    """Run a single query through the agent."""
    from ct.agent.runner import run_query

    context = context or {}
    result = run_query(query, **context)

    return result


def run_multi_agent(query: str, mode: str = "debate", context: Optional[dict] = None) -> dict:
    """Run multi-agent analysis."""
    from ct.agents.orchestrator import run_multi_agent_analysis

    return run_multi_agent_analysis(query=query, mode=mode, context=context)


def run_dmta_cycle(target: str, iterations: int = 1, num_candidates: int = 10) -> dict:
    """Run DMTA cycle."""
    from ct.campaign.dmta import run_dmta_cycle

    return run_dmta_cycle(
        target=target,
        num_candidates=num_candidates,
        iterations=iterations,
    )


def run_local_query(query: str, model: str = "llama-3-70b") -> dict:
    """Run query with local LLM."""
    from ct.local_llm.local_client import LocalLLMClient

    client = LocalLLMClient(model_name=model)
    response = client.chat(query)

    return {
        "response": response,
        "model": model,
        "mode": "local",
    }


def run_rlef_training(session_file: str) -> dict:
    """Run RLEF training from session file."""
    from ct.local_llm.rlef_trainer import RLEFTrainer

    trainer = RLEFTrainer()
    result = trainer.train(session_file)

    return result


def interactive_mode():
    """Start interactive REPL."""
    print("\n" + "="*60)
    print("CellType-Agent Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  <query>              - Run a single query")
    print("  /multi <query>       - Multi-agent analysis")
    print("  /dmta <target>       - Run DMTA cycle")
    print("  /local <query>       - Use local LLM")
    print("  /feedback <rating>   - Provide feedback (1-5)")
    print("  /stats               - Show session statistics")
    print("  /help                - Show this help")
    print("  /exit                - Exit")
    print()

    from ct.session_logging import SessionLogger
    session_logger = SessionLogger()

    while True:
        try:
            user_input = input("\nct> ").strip()

            if not user_input:
                continue

            if user_input == "/exit":
                print("Goodbye!")
                break

            elif user_input == "/help":
                print("\nAvailable commands...")
                continue

            elif user_input == "/stats":
                stats = session_logger.get_stats()
                print(f"\nSession Statistics:")
                print(f"  Total sessions: {stats['total_sessions']}")
                print(f"  Training ready: {stats['training_ready']}")
                continue

            elif user_input.startswith("/multi "):
                query = user_input[7:]
                print("\nRunning multi-agent analysis...")
                result = run_multi_agent(query)
                print(f"\n{result['summary']}")

            elif user_input.startswith("/dmta "):
                target = user_input[6:]
                print(f"\nRunning DMTA cycle for: {target[:50]}...")
                result = run_dmta_cycle(target)
                print(f"\n{result['summary']}")

            elif user_input.startswith("/local "):
                query = user_input[7:]
                print("\nRunning with local LLM...")
                result = run_local_query(query)
                print(f"\n{result['response'][:500]}...")

            elif user_input.startswith("/feedback "):
                try:
                    rating = int(user_input.split()[1])
                    session_logger.add_feedback(rating=rating)
                    print(f"\nThank you for your feedback ({rating}/5)!")
                except (ValueError, IndexError):
                    print("Usage: /feedback <1-5>")

            else:
                # Regular query
                print("\nProcessing...")
                result = run_single_query(user_input)
                print(f"\n{result.get('summary', result.get('response', 'No result'))}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit.")
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CellType-Agent: AI-powered drug discovery assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ct "What drugs target KRAS?"
  ct --mode multi-agent "Design a KRAS inhibitor"
  ct --dmta --target "KRAS_G12C" --iterations 3
  ct --local --model llama-3-70b "Analyze compound"
  ct --interactive
        """,
    )

    # Positional argument
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to process",
    )

    # Mode options
    parser.add_argument(
        "--mode", "-m",
        choices=["single", "multi-agent", "dmta", "local"],
        default="single",
        help="Execution mode",
    )

    # Multi-agent options
    parser.add_argument(
        "--agent-mode",
        choices=["sequential", "parallel", "debate"],
        default="debate",
        help="Multi-agent orchestration mode",
    )

    # DMTA options
    parser.add_argument(
        "--dmta",
        action="store_true",
        help="Run DMTA cycle",
    )
    parser.add_argument(
        "--target",
        help="Target sequence for DMTA",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of DMTA iterations",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Candidates per DMTA cycle",
    )

    # Local LLM options
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local LLM",
    )
    parser.add_argument(
        "--model",
        default="llama-3-70b",
        help="Local model to use",
    )

    # RLEF options
    parser.add_argument(
        "--rlef-train",
        help="Train RLEF from session file",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file for results",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # Knowledge graph options
    parser.add_argument(
        "--load-drkg",
        action="store_true",
        help="Load DRKG into Neo4j",
    )

    # Session options
    parser.add_argument(
        "--session-id",
        help="Session ID for logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup environment
    setup_environment()

    # Handle special commands
    if args.load_drkg:
        from ct.knowledge_graph import DRKGLoader
        loader = DRKGLoader()
        stats = loader.load()
        print(f"\nDRKG loaded: {stats.total_entities} entities, {stats.total_relations} relations")
        return 0

    if args.rlef_train:
        result = run_rlef_training(args.rlef_train)
        print(f"\nRLEF training complete: {result}")
        return 0

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return 0

    # Require query for non-interactive modes
    if not args.query:
        if args.dmta and args.target:
            args.query = f"DMTA cycle for {args.target}"
            args.mode = "dmta"
        else:
            parser.print_help()
            return 1

    # Execute based on mode
    result = {}
    context = {
        "session_id": args.session_id,
    }

    try:
        if args.mode == "single" or (not args.dmta and not args.local):
            if args.mode == "multi-agent" or "--multi" in sys.argv:
                result = run_multi_agent(args.query, args.agent_mode, context)
            else:
                result = run_single_query(args.query, context)

        elif args.mode == "multi-agent":
            result = run_multi_agent(args.query, args.agent_mode, context)

        elif args.mode == "dmta" or args.dmta:
            target = args.target or args.query
            result = run_dmta_cycle(target, args.iterations, args.candidates)

        elif args.mode == "local" or args.local:
            result = run_local_query(args.query, args.model)

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        result = {"error": str(e)}

    # Output results
    if args.format == "json":
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result.get("summary", result.get("response", str(result)))

    print(f"\n{output}")

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())