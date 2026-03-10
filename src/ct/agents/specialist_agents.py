"""
Specialist Agents for CellType-Agent Multi-Agent System.

Implements domain-specific agents:
- ChemistAgent: Molecular design and optimization
- BiologistAgent: Target and pathway analysis
- ToxicologistAgent: Safety and off-target assessment
- StatisticianAgent: Data analysis and validation
"""

import logging
from typing import Optional

from ct.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentMessage,
    AgentResponse,
)

logger = logging.getLogger("ct.agents.specialists")


class ChemistAgent(BaseAgent):
    """
    Specialist agent for chemistry and molecular design.

    Expertise:
    - Molecular structure analysis
    - Drug-like property optimization
    - Chemical synthesis planning
    - Structure-activity relationships
    """

    SYSTEM_PROMPT = """You are an expert medicinal chemist with 20+ years of drug discovery experience.

Your expertise includes:
- Small molecule design and optimization
- ADMET property prediction and optimization
- Structure-activity relationship (SAR) analysis
- Chemical synthesis feasibility assessment
- Lead optimization strategies
- Fragment-based drug design
- PROTAC and molecular glue design

When analyzing compounds:
1. Assess drug-likeness (Lipinski's Rule of 5, Veber rules)
2. Identify structural alerts and potential liabilities
3. Propose specific modifications to improve properties
4. Consider synthetic accessibility
5. Evaluate novelty and patentability

Your responses should be:
- Specific with concrete structural suggestions
- Quantitative when possible (IC50, logP, etc.)
- Practical with synthetic considerations
- Critical of potential issues

Available tools:
- admet.predict: Predict ADMET properties
- boltz2.predict_affinity: Predict binding affinity
- generative.suggest_mutations: Suggest chemical modifications
- generative.design_binder: Design new binders

Always provide:
1. Clear assessment of the compound/target
2. Specific recommendations with rationale
3. Confidence level in your analysis
4. Potential risks or concerns"""

    def __init__(self):
        super().__init__(AgentRole.CHEMIST, model="claude-sonnet-4-6")

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def get_available_tools(self) -> list[str]:
        return [
            "admet.predict",
            "admet.batch_predict",
            "boltz2.predict_affinity",
            "boltz2.virtual_screen",
            "generative.suggest_mutations",
            "generative.optimize_binder",
            "chemistry.search_chEMBL",
            "chemistry.smiles_to_3d",
        ]

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """Perform chemistry analysis."""
        query = context.get("query", "")
        compound = context.get("compound_smiles", "")
        target = context.get("target", "")

        tool_results = []
        findings = {}

        # Run ADMET if compound provided
        if compound:
            admet_result = self.call_tool("admet.predict", smiles=compound)
            tool_results.append(admet_result)
            findings["admet"] = admet_result

            if admet_result.get("critical_issues"):
                findings["safety_concerns"] = admet_result["critical_issues"]

        # Predict affinity if target provided
        if target and compound:
            affinity_result = self.call_tool(
                "boltz2.predict_affinity",
                protein_sequence=target,
                ligand_smiles=compound,
            )
            tool_results.append(affinity_result)
            findings["affinity"] = affinity_result

        # Generate analysis
        response = self.generate_response(query, workspace)
        confidence = self.calculate_confidence(findings, tool_results)

        message = AgentMessage(
            agent_role=self.role,
            content=response,
            confidence=confidence,
            tool_calls=[{"tool": r.get("tool", "unknown")} for r in tool_results],
            findings=findings,
        )

        issues = []
        if findings.get("safety_concerns"):
            issues.extend([f"ADMET: {i}" for i in findings["safety_concerns"]])

        return AgentResponse(
            success=True,
            message=message,
            issues_found=issues,
            suggested_actions=self._extract_actions(response),
        )

    def _extract_actions(self, response: str) -> list[str]:
        """Extract suggested actions from response."""
        actions = []
        # Simple extraction - look for action keywords
        action_keywords = ["recommend", "suggest", "propose", "should", "consider"]
        for line in response.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in action_keywords):
                actions.append(line)
        return actions[:5]  # Limit to 5 actions


class BiologistAgent(BaseAgent):
    """
    Specialist agent for biology and target analysis.

    Expertise:
    - Target biology and disease mechanisms
    - Pathway analysis
    - Biomarker identification
    - Cell type analysis
    - Expression data interpretation
    """

    SYSTEM_PROMPT = """You are an expert molecular biologist with deep knowledge of drug discovery biology.

Your expertise includes:
- Target validation and disease biology
- Signaling pathway analysis
- Gene expression and regulation
- Cell type-specific responses
- Biomarker discovery and validation
- Mechanism of action studies
- In vitro/in vivo model systems

When analyzing biological questions:
1. Identify relevant pathways and mechanisms
2. Assess target-disease connection strength
3. Consider tissue/cell type specificity
4. Evaluate potential resistance mechanisms
5. Identify predictive biomarkers

Your responses should:
- Connect molecular mechanisms to phenotypic outcomes
- Reference specific genes, proteins, and pathways
- Consider genetic variations and mutations
- Assess biological feasibility of proposed interventions
- Identify knowledge gaps and uncertainties

Available tools:
- knowledge.graphrag_query: Query biological knowledge graph
- knowledge.get_gene_diseases: Get gene-disease associations
- knowledge.get_drug_targets: Get drug target information
- structure.analyze_h5ad: Analyze single-cell data
- structure.extract_expression: Extract gene expression

Always provide:
1. Clear biological rationale
2. Supporting evidence from literature/databases
3. Alternative explanations or mechanisms
4. Experimental validation suggestions"""

    def __init__(self):
        super().__init__(AgentRole.BIOLOGIST, model="claude-sonnet-4-6")

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def get_available_tools(self) -> list[str]:
        return [
            "knowledge.graphrag_query",
            "knowledge.get_gene_diseases",
            "knowledge.get_drug_targets",
            "knowledge.find_path",
            "structure.analyze_h5ad",
            "structure.extract_expression",
            "expression.get_pathway_genes",
        ]

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """Perform biology analysis."""
        query = context.get("query", "")
        gene = context.get("gene", "")
        disease = context.get("disease", "")
        target = context.get("target", "")

        tool_results = []
        findings = {}

        # Query knowledge graph for relevant info
        if gene or target or disease:
            search_term = gene or target or disease
            kg_result = self.call_tool(
                "knowledge.graphrag_query",
                query=f"What are the biological roles and disease associations of {search_term}?",
            )
            tool_results.append(kg_result)
            findings["knowledge_graph"] = kg_result

        # Get gene-disease associations
        if gene:
            gd_result = self.call_tool("knowledge.get_gene_diseases", gene_name=gene)
            tool_results.append(gd_result)
            findings["gene_diseases"] = gd_result

        # Generate analysis
        response = self.generate_response(query, workspace)
        confidence = self.calculate_confidence(findings, tool_results)

        message = AgentMessage(
            agent_role=self.role,
            content=response,
            confidence=confidence,
            tool_calls=[{"tool": r.get("tool", "unknown")} for r in tool_results],
            findings=findings,
        )

        return AgentResponse(
            success=True,
            message=message,
            suggested_actions=self._extract_actions(response),
        )

    def _extract_actions(self, response: str) -> list[str]:
        """Extract suggested actions from response."""
        actions = []
        action_keywords = ["recommend", "suggest", "validate", "investigate", "consider"]
        for line in response.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in action_keywords):
                actions.append(line)
        return actions[:5]


class ToxicologistAgent(BaseAgent):
    """
    Specialist agent for toxicology and safety assessment.

    Expertise:
    - Safety profiling and risk assessment
    - Off-target effect prediction
    - Organ toxicity evaluation
    - Drug-drug interactions
    - Regulatory safety requirements
    """

    SYSTEM_PROMPT = """You are an expert toxicologist specializing in drug safety assessment.

Your expertise includes:
- Preclinical safety evaluation
- Off-target effect prediction
- Organ-specific toxicity (liver, heart, kidney, CNS)
- Drug-drug interaction assessment
- Genotoxicity and carcinogenicity
- Safety pharmacology
- Regulatory toxicology (FDA, EMA guidelines)

When assessing compound safety:
1. Evaluate all ADMET endpoints critically
2. Identify structural alerts and toxicophores
3. Assess drug-drug interaction potential (CYP inhibition/induction)
4. Consider organ-specific toxicity mechanisms
5. Evaluate safety margins relative to efficacy

Critical safety concerns to flag:
- hERG inhibition (cardiac arrhythmia risk)
- Hepatotoxicity signals (DILI)
- Genotoxicity (Ames mutagenicity)
- QT prolongation risk
- CYP3A4 inhibition (drug-drug interactions)
- BBB penetration for non-CNS drugs
- Reactive metabolite formation

Your role is to be the SAFETY CRITIC:
- Challenge optimistic assessments
- Identify worst-case scenarios
- Recommend risk mitigation strategies
- Flag compounds needing additional testing

When you identify safety issues:
- Specify the severity (critical/major/moderate/minor)
- Explain the mechanism/risk
- Suggest structural modifications if applicable
- Recommend additional safety assays

BE CONSERVATIVE - patient safety is paramount."""

    def __init__(self):
        super().__init__(AgentRole.TOXICOLOGIST, model="claude-sonnet-4-6")

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def get_available_tools(self) -> list[str]:
        return [
            "admet.predict",
            "validation.validate_protein",
            "validation.predict_aggregation",
            "validation.predict_immunogenicity",
            "knowledge.get_drug_side_effects",
        ]

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """Perform safety analysis - CRITIC role."""
        query = context.get("query", "")
        compound = context.get("compound_smiles", "")

        tool_results = []
        findings = {}
        issues_found = []

        # Always run comprehensive ADMET
        if compound:
            admet_result = self.call_tool("admet.predict", smiles=compound)
            tool_results.append(admet_result)
            findings["admet"] = admet_result

            # Extract critical issues
            if admet_result.get("critical_issues"):
                for issue in admet_result["critical_issues"]:
                    issues_found.append(f"CRITICAL: {issue.get('endpoint', 'Unknown')} - {issue.get('recommendation', '')}")

            # Check flags
            if admet_result.get("flags"):
                for flag in admet_result["flags"]:
                    if "HIGH RISK" in flag:
                        issues_found.append(flag)

        # Review other agents' findings for safety concerns
        if workspace.get("findings"):
            for role, finding in workspace["findings"].items():
                if role != self.role.value:
                    # Check for unaddressed safety concerns
                    if finding.get("affinity"):
                        # Strong binders may have off-target risk
                        pass

        # Generate safety assessment
        response = self.generate_response(query, workspace)
        confidence = self.calculate_confidence(findings, tool_results)

        # Toxicologist confidence is typically lower (conservative)
        confidence = confidence * 0.9

        message = AgentMessage(
            agent_role=self.role,
            content=response,
            confidence=confidence,
            tool_calls=[{"tool": r.get("tool", "unknown")} for r in tool_results],
            findings=findings,
        )

        return AgentResponse(
            success=len(issues_found) == 0,
            message=message,
            issues_found=issues_found,
            suggested_actions=self._extract_safety_actions(response, issues_found),
        )

    def _extract_safety_actions(self, response: str, issues: list[str]) -> list[str]:
        """Extract safety-focused actions."""
        actions = issues.copy()
        action_keywords = ["avoid", "monitor", "test", "modify", "redesign"]
        for line in response.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in action_keywords):
                actions.append(line)
        return actions[:8]


class StatisticianAgent(BaseAgent):
    """
    Specialist agent for data analysis and validation.

    Expertise:
    - Statistical analysis of experimental data
    - Study design and power calculations
    - Data quality assessment
    - Result validation
    """

    SYSTEM_PROMPT = """You are an expert biostatistician with experience in pharmaceutical research.

Your expertise includes:
- Statistical analysis of preclinical and clinical data
- Study design and power calculations
- Multiple testing correction
- Dose-response modeling
- Survival analysis
- Biomarker validation statistics
- Machine learning model evaluation

When analyzing data:
1. Assess data quality and completeness
2. Choose appropriate statistical tests
3. Consider multiple testing corrections
4. Evaluate effect sizes, not just p-values
5. Report confidence intervals

For experimental design:
- Calculate required sample sizes
- Suggest appropriate controls
- Identify potential confounders
- Recommend randomization strategies

Quality checks:
- Look for outliers and batch effects
- Check distribution assumptions
- Verify data provenance
- Assess reproducibility

Your role is to ensure RIGOR:
- Challenge underpowered studies
- Identify statistical issues
- Suggest better experimental designs
- Validate conclusions from data

Report statistics in standard format:
- Effect size with confidence intervals
- P-values with correction method
- Sample sizes and power
- Assumptions and limitations"""

    def __init__(self):
        super().__init__(AgentRole.STATISTICIAN, model="claude-sonnet-4-6")

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def get_available_tools(self) -> list[str]:
        return [
            "statistics.correlation",
            "statistics.ttest",
            "statistics.power_analysis",
            "statistics.dose_response",
        ]

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """Perform statistical analysis."""
        query = context.get("query", "")

        findings = {}

        # Check if other agents' findings are statistically sound
        if workspace.get("findings"):
            statistical_issues = []
            for role, finding in workspace["findings"].items():
                if finding.get("predictions"):
                    # Check if confidence intervals are provided
                    pass
            findings["statistical_review"] = statistical_issues

        response = self.generate_response(query, workspace)
        confidence = self.calculate_confidence(findings, [])

        message = AgentMessage(
            agent_role=self.role,
            content=response,
            confidence=confidence,
            findings=findings,
        )

        return AgentResponse(
            success=True,
            message=message,
            suggested_actions=self._extract_actions(response),
        )

    def _extract_actions(self, response: str) -> list[str]:
        """Extract suggested actions."""
        actions = []
        action_keywords = ["recommend", "calculate", "verify", "check", "ensure"]
        for line in response.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in action_keywords):
                actions.append(line)
        return actions[:5]