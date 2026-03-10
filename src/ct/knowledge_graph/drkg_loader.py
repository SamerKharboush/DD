"""
DRKG (Drug Repurposing Knowledge Graph) Loader.

Downloads and ingests the DRKG dataset into Neo4j.
DRKG contains 97,238 entities and 4.4M relationships across:
- Drugs, Diseases, Genes, Pathways, Side Effects, Anatomy
- Drug-target, drug-disease, gene-pathway relationships

Reference: https://github.com/gnn4dr/DRKG
"""

import gzip
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger("ct.knowledge_graph.drkg")


@dataclass
class DRKGStats:
    """Statistics for DRKG dataset."""
    total_entities: int = 0
    total_relations: int = 0
    entity_types: dict = field(default_factory=dict)
    relation_types: dict = field(default_factory=dict)
    load_time_seconds: float = 0.0


class DRKGLoader:
    """
    Downloads and loads DRKG into Neo4j database.

    Usage:
        loader = DRKGLoader(neo4j_uri="bolt://localhost:7687")
        stats = loader.load()
        print(f"Loaded {stats.total_entities} entities")
    """

    DRKG_URL = "https://github.com/gnn4dr/DRKG/raw/master/drkg.tar.gz"

    # Entity type prefixes in DRKG
    ENTITY_TYPES = {
        "Gene": "Gene",
        "Disease": "Disease",
        "Compound": "Drug",
        "Side Effect": "SideEffect",
        "Pathway": "Pathway",
        "Anatomy": "Anatomy",
        "Pharmacologic Class": "PharmacologicClass",
        "Symptom": "Symptom",
    }

    # Priority relation types for drug discovery
    PRIORITY_RELATIONS = [
        "Drug::Target",           # Drug binds to target
        "Drug::Disease",          # Drug treats disease
        "Drug::Side Effect",      # Drug causes side effect
        "Gene::Disease",          # Gene associated with disease
        "Gene::Pathway",          # Gene participates in pathway
        "Drug::Gene",             # Drug affects gene
        "Compound::Gene",         # Compound affects gene
        "BioRx::Gene",            # Biological relationship to gene
        "Contr::Indication",      # Contraindication
        "Drug::ATC",              # Drug ATC classification
    ]

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        data_dir: Optional[Path] = None,
        batch_size: int = 10000,
    ):
        """
        Initialize DRKG loader.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            data_dir: Directory to store downloaded data
            batch_size: Batch size for Neo4j inserts
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ct" / "drkg"
        self.batch_size = batch_size

        self._neo4j_driver = None
        self._entities_df: Optional[pd.DataFrame] = None
        self._relations_df: Optional[pd.DataFrame] = None

    @property
    def neo4j_driver(self):
        """Lazy-load Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
            except ImportError:
                raise ImportError(
                    "neo4j package required. Install with: pip install neo4j"
                )
        return self._neo4j_driver

    def download(self, force: bool = False) -> Path:
        """
        Download DRKG dataset.

        Args:
            force: Re-download even if exists

        Returns:
            Path to extracted DRKG directory
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)

        tar_path = self.data_dir / "drkg.tar.gz"
        extract_dir = self.data_dir / "drkg"

        if extract_dir.exists() and not force:
            logger.info(f"DRKG already downloaded at {extract_dir}")
            return extract_dir

        logger.info(f"Downloading DRKG from {self.DRKG_URL}")
        start = time.time()

        # Download
        urlretrieve(self.DRKG_URL, tar_path)
        logger.info(f"Downloaded in {time.time() - start:.1f}s")

        # Extract
        logger.info("Extracting...")
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.data_dir)

        tar_path.unlink()
        logger.info(f"Extracted to {extract_dir}")

        return extract_dir

    def load_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load DRKG entity and relation dataframes.

        Returns:
            Tuple of (entities_df, relations_df)
        """
        drkg_dir = self.download()

        # Load entity dictionary
        entity_file = drkg_dir / "drkg" / "entity_dict.txt"
        if entity_file.exists():
            entities = []
            with open(entity_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entities.append({
                            "id": parts[0],
                            "name": parts[1] if len(parts) > 1 else parts[0],
                            "type": parts[0].split("::")[0] if "::" in parts[0] else "Unknown"
                        })
            self._entities_df = pd.DataFrame(entities)
        else:
            # Fallback: infer entities from relations
            self._entities_df = self._infer_entities_from_relations(drkg_dir)

        # Load relations/triplets
        rel_file = drkg_dir / "drkg" / "drkg.tsv"
        if not rel_file.exists():
            rel_file = drkg_dir / "drkg.tsv"

        if rel_file.exists():
            relations = []
            with open(rel_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        relations.append({
                            "head": parts[0],
                            "relation": parts[1],
                            "tail": parts[2],
                        })
            self._relations_df = pd.DataFrame(relations)
        else:
            raise FileNotFoundError(f"DRKG relations file not found at {rel_file}")

        logger.info(
            f"Loaded {len(self._entities_df)} entities, "
            f"{len(self._relations_df)} relations"
        )

        return self._entities_df, self._relations_df

    def _infer_entities_from_relations(self, drkg_dir: Path) -> pd.DataFrame:
        """Infer entities from relation triplets."""
        rel_file = drkg_dir / "drkg" / "drkg.tsv"
        entities = set()

        with open(rel_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entities.add(parts[0])
                    entities.add(parts[2])

        entity_list = []
        for e in entities:
            entity_list.append({
                "id": e,
                "name": e.split("::")[-1] if "::" in e else e,
                "type": e.split("::")[0] if "::" in e else "Unknown"
            })

        return pd.DataFrame(entity_list)

    def create_neo4j_constraints(self):
        """Create unique constraints for entity types."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pathway) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Anatomy) REQUIRE a.id IS UNIQUE",
        ]

        with self.neo4j_driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")

        logger.info("Created Neo4j constraints")

    def load_entities_to_neo4j(self, entities_df: Optional[pd.DataFrame] = None) -> int:
        """
        Load entities into Neo4j.

        Args:
            entities_df: Optional pre-loaded entities dataframe

        Returns:
            Number of entities loaded
        """
        if entities_df is None:
            entities_df = self._entities_df or self.load_dataframes()[0]

        self.create_neo4j_constraints()

        # Group by type for batch insertion
        entity_types = entities_df['type'].value_counts().to_dict()
        loaded = 0

        for etype, count in entity_types.items():
            type_df = entities_df[entities_df['type'] == etype]
            neo4j_type = self.ENTITY_TYPES.get(etype, "Entity")

            # Batch insert
            for i in range(0, len(type_df), self.batch_size):
                batch = type_df.iloc[i:i + self.batch_size]
                records = batch.to_dict('records')

                query = f"""
                UNWIND $records AS r
                MERGE (n:{neo4j_type} {{id: r.id}})
                SET n.name = r.name,
                    n.original_type = r.type
                """

                with self.neo4j_driver.session() as session:
                    session.run(query, records=records)

            loaded += count
            logger.info(f"Loaded {loaded}/{len(entities_df)} entities ({etype})")

        return loaded

    def load_relations_to_neo4j(self, relations_df: Optional[pd.DataFrame] = None) -> int:
        """
        Load relations into Neo4j as relationships.

        Args:
            relations_df: Optional pre-loaded relations dataframe

        Returns:
            Number of relations loaded
        """
        if relations_df is None:
            relations_df = self._relations_df or self.load_dataframes()[1]

        loaded = 0

        for i in range(0, len(relations_df), self.batch_size):
            batch = relations_df.iloc[i:i + self.batch_size]
            records = batch.to_dict('records')

            # Generic relationship creation
            query = """
            UNWIND $records AS r
            MATCH (h {id: r.head})
            MATCH (t {id: r.tail})
            WITH h, t, r
            CALL apoc.create.relationship(h, r.relation, {}, t) YIELD rel
            RETURN count(rel)
            """

            try:
                with self.neo4j_driver.session() as session:
                    result = session.run(query, records=records)
                    loaded += result.single()[0] if result.single() else 0
            except Exception as e:
                # Fallback without APOC
                for record in records:
                    self._create_single_relation(record)
                    loaded += 1

            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Loaded {loaded}/{len(relations_df)} relations")

        return loaded

    def _create_single_relation(self, record: dict):
        """Create a single relationship without APOC."""
        query = """
        MATCH (h {id: $head})
        MATCH (t {id: $tail})
        MERGE (h)-[r:RELATES_TO]->(t)
        SET r.type = $relation
        """

        with self.neo4j_driver.session() as session:
            session.run(query, **record)

    def load(self, force_download: bool = False) -> DRKGStats:
        """
        Full load process: download, parse, and ingest into Neo4j.

        Args:
            force_download: Force re-download of DRKG

        Returns:
            Statistics about loaded data
        """
        start = time.time()

        # Download and load
        drkg_dir = self.download(force=force_download)
        entities_df, relations_df = self.load_dataframes()

        # Create constraints
        self.create_neo4j_constraints()

        # Load entities
        logger.info("Loading entities into Neo4j...")
        n_entities = self.load_entities_to_neo4j(entities_df)

        # Load relations
        logger.info("Loading relations into Neo4j...")
        n_relations = self.load_relations_to_neo4j(relations_df)

        load_time = time.time() - start

        stats = DRKGStats(
            total_entities=n_entities,
            total_relations=n_relations,
            entity_types=entities_df['type'].value_counts().to_dict(),
            relation_types=relations_df['relation'].value_counts().to_dict(),
            load_time_seconds=load_time,
        )

        logger.info(
            f"DRKG load complete: {n_entities} entities, {n_relations} relations "
            f"in {load_time:.1f}s"
        )

        return stats

    def verify_load(self) -> dict:
        """Verify the Neo4j load was successful."""
        with self.neo4j_driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]

            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]

            # Sample query
            sample_result = session.run("""
                MATCH (g:Gene)-[r]-(d:Disease)
                RETURN g.name, type(r), d.name
                LIMIT 3
            """)
            samples = [dict(r) for r in sample_result]

        return {
            "node_count": node_count,
            "relationship_count": rel_count,
            "sample_queries": samples,
            "expected_nodes": 97238,
            "expected_rels": 4441249,
            "node_load_pct": node_count / 97238 * 100 if node_count else 0,
            "rel_load_pct": rel_count / 4441249 * 100 if rel_count else 0,
        }

    def close(self):
        """Close Neo4j connection."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None