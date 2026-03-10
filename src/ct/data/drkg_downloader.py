"""
DRKG Data Downloader and Loader.

Downloads and prepares DRKG (Drug Repurposing Knowledge Graph) data.
"""

import gzip
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.data.drkg")


class DRKGDownloader:
    """
    Downloads and extracts DRKG data.

    DRKG is a biological knowledge graph with:
    - 5.8M triplets
    - 58K drug-disease relationships
    - 2.9M gene-gene interactions
    - 8.8M drug-gene relationships

    Usage:
        downloader = DRKGDownloader()
        downloader.download()
        downloader.load_to_neo4j()
    """

    DRKG_URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
    DRKG_ANNOTATIONS_URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/embeddings/"

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DRKG downloader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ct" / "data" / "drkg"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.downloaded = False
        self.extracted = False

    def download(self, force: bool = False) -> Path:
        """
        Download DRKG data.

        Args:
            force: Force re-download

        Returns:
            Path to downloaded file
        """
        download_path = self.data_dir / "drkg.tar.gz"

        if download_path.exists() and not force:
            logger.info(f"DRKG already downloaded: {download_path}")
            self.downloaded = True
            return download_path

        logger.info(f"Downloading DRKG from {self.DRKG_URL}...")
        logger.info("This may take a few minutes (file is ~300MB)")

        try:
            urllib.request.urlretrieve(self.DRKG_URL, download_path)
            self.downloaded = True
            logger.info(f"Downloaded to {download_path}")
            return download_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def extract(self, force: bool = False) -> Path:
        """
        Extract DRKG data.

        Args:
            force: Force re-extraction

        Returns:
            Path to extracted directory
        """
        if not self.downloaded:
            self.download()

        extract_dir = self.data_dir / "drkg"

        if extract_dir.exists() and not force:
            logger.info(f"DRKG already extracted: {extract_dir}")
            self.extracted = True
            return extract_dir

        logger.info("Extracting DRKG data...")

        download_path = self.data_dir / "drkg.tar.gz"

        try:
            import tarfile
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(self.data_dir)

            self.extracted = True
            logger.info(f"Extracted to {extract_dir}")
            return extract_dir

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    def get_triplets(self) -> list[tuple[str, str, str]]:
        """
        Get DRKG triplets.

        Returns:
            List of (head, relation, tail) triplets
        """
        if not self.extracted:
            self.extract()

        triplets_file = self.data_dir / "drkg" / "drkg.tsv"

        if not triplets_file.exists():
            # Try alternative paths
            for f in (self.data_dir / "drkg").glob("**/*.tsv"):
                if "drkg" in f.name.lower() or "triplet" in f.name.lower():
                    triplets_file = f
                    break

        if not triplets_file.exists():
            logger.warning(f"Triplets file not found at {triplets_file}")
            return []

        triplets = []
        with open(triplets_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    triplets.append((parts[0], parts[1], parts[2]))

        logger.info(f"Loaded {len(triplets)} triplets")
        return triplets

    def get_entity_types(self) -> dict[str, list[str]]:
        """
        Get entity type mappings.

        Returns:
            Dictionary mapping entity types to entity IDs
        """
        if not self.extracted:
            self.extract()

        entities = {}

        # Look for entity mapping files
        for f in (self.data_dir / "drkg").glob("**/*entity*.tsv"):
            try:
                with open(f) as fp:
                    for line in fp:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            entity_type = parts[0].split("::")[0] if "::" in parts[0] else "Unknown"
                            if entity_type not in entities:
                                entities[entity_type] = []
                            entities[entity_type].append(parts[0])
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        return entities

    def load_to_neo4j(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        batch_size: int = 10000,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Load DRKG into Neo4j.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            batch_size: Batch size for loading
            limit: Maximum number of triplets to load

        Returns:
            Loading statistics
        """
        from ct.knowledge_graph.neo4j_client import Neo4jClient, get_neo4j_client

        client = get_neo4j_client()

        triplets = self.get_triplets()

        if limit:
            triplets = triplets[:limit]

        logger.info(f"Loading {len(triplets)} triplets into Neo4j...")

        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": 0,
        }

        # Create constraint for uniqueness
        try:
            client.run_query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            )
        except Exception:
            pass

        # Process in batches
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]

            for head, relation, tail in batch:
                try:
                    # Extract entity type
                    head_type = head.split("::")[0] if "::" in head else "Entity"
                    tail_type = tail.split("::")[0] if "::" in tail else "Entity"

                    # Clean names
                    head_name = head.split("::")[-1] if "::" in head else head
                    tail_name = tail.split("::")[-1] if "::" in tail else tail
                    relation_name = relation.split("::")[-1] if "::" in relation else relation

                    # Create nodes and relationship
                    query = f"""
                    MERGE (h:{head_type} {{id: $head_id, name: $head_name}})
                    MERGE (t:{tail_type} {{id: $tail_id, name: $tail_name}})
                    MERGE (h)-[r:{self._sanitize_relation(relation_name)}]->(t)
                    SET r.type = $relation_type
                    """

                    client.run_query(
                        query,
                        params={
                            "head_id": head,
                            "head_name": head_name,
                            "tail_id": tail,
                            "tail_name": tail_name,
                            "relation_type": relation,
                        }
                    )

                    stats["nodes_created"] += 2
                    stats["relationships_created"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.debug(f"Failed to load triplet: {e}")

            if (i + batch_size) % 100000 == 0:
                logger.info(f"Progress: {i + batch_size}/{len(triplets)}")

        logger.info(f"Loading complete: {stats}")
        return stats

    def _sanitize_relation(self, relation: str) -> str:
        """Sanitize relation name for Neo4j."""
        # Remove special characters
        sanitized = relation.replace(" ", "_").replace("-", "_")
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "R_" + sanitized
        return sanitized[:50]  # Limit length

    def get_stats(self) -> dict:
        """Get DRKG statistics."""
        stats = {
            "downloaded": self.downloaded,
            "extracted": self.extracted,
            "data_dir": str(self.data_dir),
        }

        if self.extracted:
            triplets = self.get_triplets()
            stats["num_triplets"] = len(triplets)

            entities = self.get_entity_types()
            stats["entity_types"] = {k: len(v) for k, v in entities.items()}

        return stats


def download_drkg(data_dir: Optional[Path] = None) -> Path:
    """
    Convenience function to download DRKG.

    Args:
        data_dir: Optional data directory

    Returns:
        Path to downloaded data
    """
    downloader = DRKGDownloader(data_dir)
    return downloader.extract()