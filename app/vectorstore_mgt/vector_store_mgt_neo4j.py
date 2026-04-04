import logging

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
# from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j import Neo4jVector
# from src.chat_agent.graph_utils.graph_transformer import extract_graph_triples
# from src.chat_agent.graph_utils.graph_ingestion import ingest_triples_to_neo4j
from neo4j import GraphDatabase

from src.chat_agent.vector_store_mgt.vector_store_mgt import VectorStoreMgt

logger = logging.getLogger(__name__)


class VectorStoreMgtNeo4j(VectorStoreMgt):
    def __init__(
            self,
            vector_store_dir,
            embedding_function,
            neo4j_url,
            username,
            password,
            llm,
            **kwargs,
    ):
        super().__init__(vector_store_dir, embedding_function, **kwargs)
        self.neo4j_url = neo4j_url
        self.username = username
        self.password = password

        self.vector_index = Neo4jVector(
            url=neo4j_url,
            username=username,
            password=password,
            embedding=self.embedding_function,
            database=kwargs.get("database", "neo4j"),
        )
        self.llm = llm

        # Auto-create vector index if needed
        try:
            embedding_dimension = len(self.embedding_function.embed_query("dummy test"))
            self._ensure_vector_index(dimension=embedding_dimension)
        except Exception as e:
            logger.warning(f"Could not verify/create vector index: {e}")

    def get_latest_vector_store(self):
        return self.vector_index

    def batch_update(
        self, new_documents: list[Document], deduplicate: bool = True, **kwargs
    ):
        unique_docs = new_documents
        if deduplicate:
            logger.info("Deduplication for Neo4j vector store not yet implemented.")
            # TODO: Implement vector-based dedup via Cypher if needed

        logger.info(f"Indexing {len(unique_docs)} documents into Neo4j vector store.")
        self.vector_index.add_documents(unique_docs)

        # # Option 1 (we do not use this)
        # # Graph RAG: extract and ingest triples
        # logger.info("Extracting graph triples from chunks.")
        # triples = extract_graph_triples(unique_docs)
        # logger.info(f"Extracted {len(triples)} triples.")
        # ingest_triples_to_neo4j(triples, self.neo4j_url, self.username, self.password)

        # Option 2 (we use this built in with langhchain_neo4j)
        # GRAPH INGESTION USING GraphDocument ---
        logger.info("🔍 Converting chunks to GraphDocuments using LLMGraphTransformer.")
        transformer = LLMGraphTransformer(
            llm=self.llm,
            # allowed_nodes = [
            #     "Subfund", "ShareClass", "Organization", "Person", "Role",
            #     "Document", "Clause", "Fee", "Threshold", #"LegalTerm",
            #     "Amount", "Jurisdiction", "Asset"
            # ],
            allowed_nodes=[
                "Organization",
                "Fund",
                "ShareClass",
                "LegalDocument",
                "Transaction",
                "Note",
                "Person",
                "Role",
                "Date",
                "Fee",
            ],
            # allowed_relationships = [
            #     "HAS_CLAUSE",
            #     "MANAGED_BY", "HAS_ORIGINATOR",
            #     "ACTED_AS", "REPRESENTS",
            #     "CHARGES_FEE", what about nodes
            #     "HAS_THRESHOLD",
            #     "HAS_SHARECLASS", "INCLUDES_ASSET", "REFERS_TO"
            # ],
            allowed_relationships=[
                "HAS_PARTY",
                "HAS_ROLE",
                "HELD_BY",
                "PAYS_FEE",
                "ISSUED_BY",
                "GUARANTEED_BY",
                "COLLATERALIZED_BY",
                "PART_OF",
                "EFFECTIVE_ON",
            ],
            node_properties=False,  # [
            #     "name", "type", "clause_number", "title", "text",
            #     "rate", "value", "currency", "unit", "role",
            #     "jurisdiction", "date", "doc_type", "document_id",
            #     "term", "definition", "hedged", "distribution_type"
            # ],
            relationship_properties=False,  # [
            #     "amount", "currency", "rate", "unit", "frequency",
            #     "start_date", "end_date", "confidence",
            #     "source_clause", "document_id", "applies_to", "trigger_event"
            # ],
            strict_mode=True,
            # additional_instructions = "Extract only clearly stated and unambiguous relationships. Avoid speculative or overly granular nodes. Focus on named and structurally relevant entities. Do not attempt to model clauses, terms, or subjective interpretations. Omit relationships if uncertain."
        )
        graph_docs = transformer.convert_to_graph_documents(unique_docs)
        logger.info(f"Extracted {len(graph_docs)} GraphDocuments.")

        logger.info(
            "🌐 Ingesting graph documents into Neo4j using add_graph_documents()."
        )
        graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.username,
            password=self.password,
        )
        graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
        logger.info("✅ Graph ingestion complete.")

    def _ensure_vector_index(
        self,
        label: str = "Chunk",
        property: str = "embedding",
        index_name: str = "vector",
        dimension: int = 1536,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Ensures the required vector index exists. Uses SHOW INDEXES to check if the index
        is present. If not, attempts to create it. Ignores 'already exists' errors.
        """
        logger.info(f"Ensuring vector index '{index_name}' exists...")

        driver = GraphDatabase.driver(
            self.neo4j_url, auth=(self.username, self.password)
        )
        try:
            with driver.session(database=self.vector_index._database) as session:
                try:
                    # Check if the index exists using SHOW INDEXES
                    result = session.run(
                        """
                        SHOW INDEXES YIELD name, type
                        WHERE type = 'VECTOR'
                        RETURN name
                    """
                    )
                    existing_indexes = [row["name"] for row in result]
                    if index_name in existing_indexes:
                        logger.info(f"Vector index '{index_name}' already exists.")
                        return
                    else:
                        logger.info(
                            f"Vector index '{index_name}' not found. Creating..."
                        )

                except Exception as check_error:
                    logger.warning(
                        f"Index existence check via SHOW INDEXES failed, proceeding to creation anyway: {check_error}",
                        exc_info=True,
                    )

                # Try to create the index unconditionally
                try:
                    session.run(
                        f"""
                        CALL db.index.vector.createNodeIndex(
                            '{index_name}',
                            '{label}',
                            '{property}',
                            {dimension},
                            '{distance_metric.upper()}'
                        )
                    """
                    )
                    logger.info(f"Vector index '{index_name}' successfully created.")
                except Exception as create_error:
                    if "already exists" in str(create_error):
                        logger.info(
                            f"Vector index '{index_name}' already exists (during creation)."
                        )
                    else:
                        logger.error(
                            f"Failed to create vector index '{index_name}': {create_error}",
                            exc_info=True,
                        )
                        raise

        finally:
            driver.close()
