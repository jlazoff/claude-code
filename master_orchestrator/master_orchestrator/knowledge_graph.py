"""ArangoDB-based Knowledge Graph for Master Orchestrator."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

import structlog
from arango import ArangoClient
from arango.database import Database
from arango.collection import Collection
from pydantic import BaseModel, Field

from .config import ArangoDBConfig

logger = structlog.get_logger()


class GraphNode(BaseModel):
    """Base graph node model."""
    
    _key: Optional[str] = Field(default=None)
    _id: Optional[str] = Field(default=None)
    _rev: Optional[str] = Field(default=None)
    
    node_type: str = Field(description="Type of node")
    name: str = Field(description="Node name")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectNode(GraphNode):
    """Project/Repository node."""
    
    node_type: str = Field(default="project")
    path: str = Field(description="Project path")
    technologies: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    last_analysis: Optional[datetime] = Field(default=None)


class AgentNode(GraphNode):
    """Agent node."""
    
    node_type: str = Field(default="agent")
    agent_type: str = Field(description="Type of agent")
    status: str = Field(default="idle")
    current_task: Optional[str] = Field(default=None)
    capabilities: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class HardwareNode(GraphNode):
    """Hardware node."""
    
    node_type: str = Field(default="hardware")
    hardware_type: str = Field(description="Type of hardware")
    specifications: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="online")
    current_load: float = Field(default=0.0)
    ip_address: Optional[str] = Field(default=None)


class TaskNode(GraphNode):
    """Task node."""
    
    node_type: str = Field(default="task")
    task_type: str = Field(description="Type of task")
    status: str = Field(default="pending")
    priority: int = Field(default=5)
    assigned_agent: Optional[str] = Field(default=None)
    assigned_hardware: Optional[str] = Field(default=None)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Base graph edge model."""
    
    _key: Optional[str] = Field(default=None)
    _id: Optional[str] = Field(default=None)
    _rev: Optional[str] = Field(default=None)
    _from: str = Field(description="Source node ID")
    _to: str = Field(description="Target node ID")
    
    edge_type: str = Field(description="Type of relationship")
    weight: float = Field(default=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeGraph:
    """
    ArangoDB-based Knowledge Graph for the Master Orchestrator.
    
    Manages relationships between projects, agents, hardware, tasks,
    and system metrics in a unified graph structure.
    """
    
    def __init__(self, config: ArangoDBConfig):
        self.config = config
        self.logger = structlog.get_logger("knowledge_graph")
        
        self.client: Optional[ArangoClient] = None
        self.database: Optional[Database] = None
        
        # Collections
        self.nodes: Optional[Collection] = None
        self.edges: Optional[Collection] = None
        self.metrics: Optional[Collection] = None
        self.workflows: Optional[Collection] = None
        
        self.is_connected = False
    
    async def initialize(self) -> None:
        """Initialize the knowledge graph database."""
        self.logger.info("Initializing Knowledge Graph")
        
        try:
            # Connect to ArangoDB
            self.client = ArangoClient(
                hosts=f"{self.config.protocol}://{self.config.host}:{self.config.port}"
            )
            
            # Connect to system database first
            sys_db = self.client.db(
                "_system",
                username=self.config.username,
                password=self.config.password
            )
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.config.database):
                sys_db.create_database(self.config.database)
                self.logger.info(f"Created database: {self.config.database}")
            
            # Connect to our database
            self.database = self.client.db(
                self.config.database,
                username=self.config.username,
                password=self.config.password
            )
            
            # Create collections
            await self._create_collections()
            
            # Create indexes
            await self._create_indexes()
            
            self.is_connected = True
            self.logger.info("Knowledge Graph initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Graph: {e}")
            raise
    
    async def _create_collections(self) -> None:
        """Create necessary collections."""
        collections = {
            "nodes": "document",
            "edges": "edge", 
            "metrics": "document",
            "workflows": "document"
        }
        
        for collection_name, collection_type in collections.items():
            if not self.database.has_collection(collection_name):
                if collection_type == "edge":
                    collection = self.database.create_collection(
                        collection_name,
                        edge=True
                    )
                else:
                    collection = self.database.create_collection(collection_name)
                
                self.logger.info(f"Created collection: {collection_name}")
            else:
                collection = self.database.collection(collection_name)
            
            # Assign to instance
            setattr(self, collection_name, collection)
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for performance."""
        if self.nodes:
            # Index on node_type for fast queries
            self.nodes.add_index({
                "type": "persistent",
                "fields": ["node_type"]
            })
            
            # Index on name for searches
            self.nodes.add_index({
                "type": "persistent", 
                "fields": ["name"]
            })
            
            # Index on created_at for time-based queries
            self.nodes.add_index({
                "type": "persistent",
                "fields": ["created_at"]
            })
        
        if self.edges:
            # Index on edge_type
            self.edges.add_index({
                "type": "persistent",
                "fields": ["edge_type"]
            })
    
    async def add_node(self, node: GraphNode) -> str:
        """Add a node to the knowledge graph."""
        if not self.nodes:
            raise RuntimeError("Knowledge graph not initialized")
        
        node_data = node.model_dump(exclude_none=True)
        node_data["updated_at"] = datetime.utcnow()
        
        result = self.nodes.insert(node_data)
        node_id = result["_id"]
        
        self.logger.debug(f"Added node: {node_id}", node_type=node.node_type)
        return node_id
    
    async def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the knowledge graph."""
        if not self.edges:
            raise RuntimeError("Knowledge graph not initialized")
        
        edge_data = edge.model_dump(exclude_none=True)
        result = self.edges.insert(edge_data)
        edge_id = result["_id"]
        
        self.logger.debug(f"Added edge: {edge_id}", edge_type=edge.edge_type)
        return edge_id
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        if not self.nodes:
            return None
        
        try:
            return self.nodes.get(node_id)
        except Exception:
            return None
    
    async def find_nodes(
        self,
        node_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        if not self.database:
            return []
        
        aql_query = "FOR node IN nodes"
        bind_vars = {}
        
        conditions = []
        if node_type:
            conditions.append("node.node_type == @node_type")
            bind_vars["node_type"] = node_type
        
        if filters:
            for key, value in filters.items():
                conditions.append(f"node.{key} == @{key}")
                bind_vars[key] = value
        
        if conditions:
            aql_query += " FILTER " + " AND ".join(conditions)
        
        aql_query += f" LIMIT {limit} RETURN node"
        
        cursor = self.database.aql.execute(aql_query, bind_vars=bind_vars)
        return list(cursor)
    
    async def find_related_nodes(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Find nodes related to a given node."""
        if not self.database:
            return []
        
        if direction == "outbound":
            aql_query = """
                FOR vertex IN 1..1 OUTBOUND @node_id edges
                FILTER @edge_type == null OR edge.edge_type == @edge_type
                RETURN vertex
            """
        elif direction == "inbound":
            aql_query = """
                FOR vertex IN 1..1 INBOUND @node_id edges
                FILTER @edge_type == null OR edge.edge_type == @edge_type
                RETURN vertex
            """
        else:  # both
            aql_query = """
                FOR vertex IN 1..1 ANY @node_id edges
                FILTER @edge_type == null OR edge.edge_type == @edge_type
                RETURN vertex
            """
        
        bind_vars = {
            "node_id": node_id,
            "edge_type": edge_type
        }
        
        cursor = self.database.aql.execute(aql_query, bind_vars=bind_vars)
        return list(cursor)
    
    async def update_system_metrics(self, status: Any) -> None:
        """Update system metrics in the knowledge graph."""
        if not self.metrics:
            return
        
        metric_data = {
            "metric_type": "system_status",
            "timestamp": datetime.utcnow(),
            "data": status.model_dump() if hasattr(status, 'model_dump') else status
        }
        
        self.metrics.insert(metric_data)
    
    async def get_pending_workflows(self) -> List[Dict[str, Any]]:
        """Get pending workflows from the knowledge graph."""
        if not self.workflows:
            return []
        
        cursor = self.workflows.find({"status": "pending"})
        return list(cursor)
    
    async def create_project_node(self, project_info: Dict[str, Any]) -> str:
        """Create a project node."""
        project_node = ProjectNode(
            name=project_info["name"],
            path=project_info["path"],
            technologies=project_info.get("technologies", []),
            capabilities=project_info.get("capabilities", []),
            metadata=project_info.get("metadata", {})
        )
        
        return await self.add_node(project_node)
    
    async def create_agent_node(self, agent_info: Dict[str, Any]) -> str:
        """Create an agent node."""
        agent_node = AgentNode(
            name=agent_info["name"],
            agent_type=agent_info["type"],
            capabilities=agent_info.get("capabilities", []),
            metadata=agent_info.get("metadata", {})
        )
        
        return await self.add_node(agent_node)
    
    async def create_hardware_node(self, hardware_info: Dict[str, Any]) -> str:
        """Create a hardware node."""
        hardware_node = HardwareNode(
            name=hardware_info["name"],
            hardware_type=hardware_info["type"],
            specifications=hardware_info.get("specifications", {}),
            ip_address=hardware_info.get("ip_address"),
            metadata=hardware_info.get("metadata", {})
        )
        
        return await self.add_node(hardware_node)
    
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a relationship between two nodes."""
        edge = GraphEdge(
            _from=from_node_id,
            _to=to_node_id,
            edge_type=relationship_type,
            properties=properties or {}
        )
        
        return await self.add_edge(edge)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get a comprehensive system overview."""
        if not self.database:
            return {}
        
        # Count nodes by type
        node_counts = {}
        for node_type in ["project", "agent", "hardware", "task"]:
            cursor = self.database.aql.execute(
                "FOR node IN nodes FILTER node.node_type == @type COLLECT WITH COUNT INTO count RETURN count",
                bind_vars={"type": node_type}
            )
            count = list(cursor)
            node_counts[node_type] = count[0] if count else 0
        
        # Get recent metrics
        cursor = self.database.aql.execute(
            "FOR metric IN metrics FILTER metric.metric_type == 'system_status' SORT metric.timestamp DESC LIMIT 1 RETURN metric"
        )
        recent_metrics = list(cursor)
        
        return {
            "node_counts": node_counts,
            "recent_metrics": recent_metrics[0] if recent_metrics else None,
            "last_updated": datetime.utcnow()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the knowledge graph connection."""
        self.logger.info("Shutting down Knowledge Graph")
        
        # Close database connection
        if self.client:
            self.client.close()
        
        self.is_connected = False
        self.logger.info("Knowledge Graph shutdown complete")