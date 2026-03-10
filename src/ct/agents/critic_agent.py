"""
Critic Agent for Adversarial Review.

The critic agent acts as an adversarial reviewer, challenging
other agents' conclusions and identifying potential issues.
This implements the Robin paradigm's adversarial critique.
"""

import logging
from typing import Optional

from ct.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentMessage,
    AgentResponse,
)

logger = logging.getLogger("ct.agents.critic")


class CriticAgent(BaseAgent):
    """
    Adversarial critic agent for quality control.

    The critic's role is to:
    - Challenge assumptions made by other agents
    - Identify potential errors and oversights
    - Provide counter-arguments
    - Ensure rigor in conclusions

    This implements the adversarial critique pattern from
    the Robin drug discovery system.
    """

    SYSTEM_PROMPT = """You are an adversarial critic agent. Your role is to CHALLENGE and VERIFY.

Your mission:
- Find weaknesses in other agents' reasoning
- Identify overlooked risks and issues
- Provide counter-arguments to conclusions
- Ensure scientific rigor

CRITICAL THINKING FRAMEWORK:

1. Assumption Analysis:
   - What assumptions are implicit?
   - Which assumptions might be wrong?
   - What evidence supports each assumption?

2. Alternative Explanations:
   - Could results be interpreted differently?
   - Are there confounding factors?
   - What else could explain the observations?

3. Missing Information:
   - What key data is unavailable?
   - What experiments weren't done?
   - What questions weren't asked?

4. Risk Identification:
   - What could go wrong?
   - What are the edge cases?
   - What are the failure modes?

5. Evidence Quality:
   - How strong is the supporting evidence?
   - Are there contradictions?
   - Is the confidence justified?

REVIEW APPROACH:

For Chemist findings:
- Check ADMET issues might be missed
- Verify affinity predictions are realistic
- Question synthetic accessibility claims

For Biologist findings:
- Challenge target-disease links
- Question pathway assumptions
- Identify alternative mechanisms

For Toxicologist findings:
- Are safety concerns adequately addressed?
- Are risk mitigations sufficient?
- What additional testing is needed?

OUTPUT FORMAT:

Your review should identify:
1. STRENGTHS: What is well-supported
2. WEAKNESSES: What needs more evidence
3. GAPS: What is missing
4. RISKS: What could go wrong
5. RECOMMENDATIONS: What to verify/redo

Be CONSTRUCTIVE but CRITICAL. Your goal is to improve the final answer,
not to be negative for negativity's sake. But do NOT hold back on
identifying real issues.

Remember: It's better to catch problems now than in the lab."""

    def __init__(self):
        super().__init__(AgentRole.CRITIC, model="claude-opus-4-6")
        self.temperature = 0.5  # Higher temperature for diverse critique

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def get_available_tools(self) -> list[str]:
        return [
            "admet.predict",
            "knowledge.graphrag_query",
            "validation.validate_protein",
        ]

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """Perform adversarial critique of workspace."""
        query = context.get("query", "")

        # Gather all findings for review
        all_findings = workspace.get("findings", {})

        issues = []
        counter_arguments = []
        verification_needed = []

        # Review each agent's findings
        for role, findings in all_findings.items():
            if role == self.role.value:
                continue

            agent_issues = self._review_agent_findings(role, findings, context)
            issues.extend(agent_issues["issues"])
            counter_arguments.extend(agent_issues["counter_arguments"])
            verification_needed.extend(agent_issues["verifications"])

        # Generate comprehensive critique
        critique = self._generate_critique(
            query=query,
            workspace=workspace,
            issues=issues,
            counter_arguments=counter_arguments,
        )

        # Calculate confidence in critique
        confidence = self._calculate_critique_confidence(issues, all_findings)

        message = AgentMessage(
            agent_role=self.role,
            content=critique,
            confidence=confidence,
            findings={
                "issues_found": issues,
                "counter_arguments": counter_arguments,
                "verification_needed": verification_needed,
            },
        )

        return AgentResponse(
            success=len(issues) == 0,
            message=message,
            issues_found=issues,
            suggested_actions=verification_needed,
        )

    def _review_agent_findings(
        self,
        role: str,
        findings: dict,
        context: dict,
    ) -> dict:
        """Review findings from a specific agent."""
        result = {
            "issues": [],
            "counter_arguments": [],
            "verifications": [],
        }

        if role == AgentRole.CHEMIST.value:
            result.update(self._critique_chemist(findings, context))
        elif role == AgentRole.BIOLOGIST.value:
            result.update(self._critique_biologist(findings, context))
        elif role == AgentRole.TOXICOLOGIST.value:
            result.update(self._critique_toxicologist(findings, context))
        elif role == AgentRole.STATISTICIAN.value:
            result.update(self._critique_statistician(findings, context))

        return result

    def _critique_chemist(self, findings: dict, context: dict) -> dict:
        """Critique chemist's findings."""
        issues = []
        counter_arguments = []
        verifications = []

        admet = findings.get("admet", {})

        # Check if ADMET was actually run
        if not admet or admet.get("error"):
            issues.append("ADMET analysis incomplete or failed")
            verifications.append("Re-run ADMET prediction")

        # Check for overlooked critical endpoints
        if admet.get("predictions"):
            critical_endpoints = ["herg_inhibitor", "bbb_permeability", "dili"]
            for endpoint in critical_endpoints:
                if endpoint not in admet.get("predictions", {}):
                    issues.append(f"Critical endpoint {endpoint} not assessed")

        # Check confidence calibration
        if findings.get("confidence", 0) > 0.9:
            issues.append("Chemist confidence seems overcalibrated")
            counter_arguments.append("Results should be validated with experimental data")

        # Check affinity claims
        affinity = findings.get("affinity", {})
        if affinity.get("predicted_affinity_nm"):
            if affinity["predicted_affinity_nm"] < 1:
                issues.append("Unrealistic sub-nanomolar prediction - verify with structural analysis")

        return {
            "issues": issues,
            "counter_arguments": counter_arguments,
            "verifications": verifications,
        }

    def _critique_biologist(self, findings: dict, context: dict) -> dict:
        """Critique biologist's findings."""
        issues = []
        counter_arguments = []
        verifications = []

        kg = findings.get("knowledge_graph", {})

        # Check if knowledge graph was queried
        if not kg or kg.get("error"):
            issues.append("Knowledge graph query incomplete")

        # Check for alternative pathways
        gene_diseases = findings.get("gene_diseases", {})
        if gene_diseases and gene_diseases.get("diseases"):
            if len(gene_diseases["diseases"]) > 10:
                counter_arguments.append(
                    "Target associated with many diseases - specificity concern"
                )

        # Check for missing context
        if not findings.get("cell_type_specificity"):
            issues.append("Cell type specificity not addressed")
            verifications.append("Validate target expression in relevant cell types")

        return {
            "issues": issues,
            "counter_arguments": counter_arguments,
            "verifications": verifications,
        }

    def _critique_toxicologist(self, findings: dict, context: dict) -> dict:
        """Critique toxicologist's findings."""
        issues = []
        counter_arguments = []
        verifications = []

        admet = findings.get("admet", {})

        # Toxicologist should be conservative
        if findings.get("confidence", 0) > 0.8:
            issues.append("Toxicologist confidence unusually high - verify conservatism")

        # Check if all issues were addressed
        if admet.get("flags"):
            unresolved = [f for f in admet["flags"] if "RESOLVED" not in f]
            if unresolved:
                issues.append(f"{len(unresolved)} ADMET flags not resolved")

        return {
            "issues": issues,
            "counter_arguments": counter_arguments,
            "verifications": verifications,
        }

    def _critique_statistician(self, findings: dict, context: dict) -> dict:
        """Critique statistician's findings."""
        issues = []
        counter_arguments = []
        verifications = []

        # Check for statistical rigor
        if not findings.get("confidence_intervals"):
            issues.append("Confidence intervals not provided")

        if not findings.get("sample_size_justification"):
            counter_arguments.append("Sample size adequacy not demonstrated")

        return {
            "issues": issues,
            "counter_arguments": counter_arguments,
            "verifications": verifications,
        }

    def _generate_critique(
        self,
        query: str,
        workspace: dict,
        issues: list[str],
        counter_arguments: list[str],
    ) -> str:
        """Generate comprehensive critique."""
        # Build critique prompt
        critique_prompt = f"""Review the multi-agent analysis for: {query}

ISSUES IDENTIFIED:
{chr(10).join(f'- {i}' for i in issues) if issues else '- No major issues identified'}

COUNTER-ARGUMENTS:
{chr(10).join(f'- {c}' for c in counter_arguments) if counter_arguments else '- None'}

Provide your adversarial critique:
1. Are there any overlooked issues?
2. Are the conclusions justified?
3. What additional verification is needed?
4. Should the analysis be redone?"""

        # Generate critique
        response = self.generate_response(query, workspace, critique_prompt)

        return response

    def _calculate_critique_confidence(
        self,
        issues: list[str],
        all_findings: dict,
    ) -> float:
        """Calculate confidence in the critique."""
        # Base confidence
        confidence = 0.7

        # More findings to review = higher confidence in critique
        num_agents = len([f for f in all_findings.values() if f])
        confidence += min(num_agents * 0.05, 0.2)

        # More issues found = lower confidence in overall analysis
        if len(issues) > 5:
            confidence = min(confidence, 0.5)

        return min(1.0, confidence)


def run_adversarial_review(
    workspace: dict,
    context: dict,
) -> AgentResponse:
    """
    Run adversarial review on workspace.

    Args:
        workspace: Shared workspace with agent findings
        context: Query context

    Returns:
        Critic agent response
    """
    critic = CriticAgent()
    return critic.analyze(context, workspace)