"""
LazyCoder Agent Orchestrator
Central coordinator for all AI agents and system operations.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

from ...core.config import get_config, LazyCoderConfig


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task to be executed by agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_type: str = ""
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None


@dataclass
class AgentInfo:
    """Information about registered agents"""
    name: str
    agent_type: str
    instance: Any
    capabilities: List[str]
    is_active: bool = True
    last_used: Optional[datetime] = None


class AgentOrchestrator:
    """
    Central orchestrator that coordinates all AI agents and manages task execution.
    Implements the main control loop for autonomous operation.
    """
    
    def __init__(self, config: LazyCoderConfig):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.is_running = False
        self.emergency_stop = False
        
        # Initialize components
        self.autonomy_controller = None
        self.memory_system = None
        self.safety_monitor = None
        
        logger.info("AgentOrchestrator initialized")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize autonomy controller
            from ..autonomy.autonomy_controller import AutonomyController
            self.autonomy_controller = AutonomyController(self.config)
            await self.autonomy_controller.initialize()
            
            # Initialize memory system (placeholder for future implementation)
            self.memory_system = None
            logger.info("Memory system placeholder initialized")
            
            # Initialize safety monitor (placeholder for future implementation)
            self.safety_monitor = None
            logger.info("Safety monitor placeholder initialized")
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}")
            raise
    
    def register_agent(self, agent_instance: Any, agent_type: str, 
                      capabilities: List[str]) -> str:
        """Register a new agent with the orchestrator"""
        agent_name = f"{agent_type}_{len(self.agents)}"
        
        agent_info = AgentInfo(
            name=agent_name,
            agent_type=agent_type,
            instance=agent_instance,
            capabilities=capabilities
        )
        
        self.agents[agent_name] = agent_info
        logger.info(f"Registered agent: {agent_name} with capabilities: {capabilities}")
        
        return agent_name
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")
            return True
        return False
    
    def create_task(self, name: str, description: str, agent_type: str, 
                   action: str, parameters: Dict[str, Any] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   dependencies: List[str] = None,
                   callback: Callable = None) -> str:
        """Create a new task"""
        task = Task(
            name=name,
            description=description,
            agent_type=agent_type,
            action=action,
            parameters=parameters or {},
            priority=priority,
            dependencies=dependencies or [],
            callback=callback
        )
        
        # Insert task based on priority
        self._insert_task_by_priority(task)
        
        logger.info(f"Created task: {task.id} - {task.name}")
        return task.id
    
    def _insert_task_by_priority(self, task: Task):
        """Insert task into queue based on priority"""
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task)
    
    async def execute_task(self, task: Task) -> bool:
        """Execute a single task"""
        try:
            # Check if emergency stop is active
            if self.emergency_stop:
                logger.warning("Emergency stop active - cancelling task execution")
                task.status = TaskStatus.CANCELLED
                return False
            
            # Check autonomy level
            if not await self.autonomy_controller.can_execute_action(
                task.action, task.parameters
            ):
                logger.info(f"Task {task.id} requires user confirmation")
                # Handle user confirmation logic here
                return False
            
            # Find appropriate agent
            agent = self._find_agent_for_task(task)
            if not agent:
                logger.error(f"No suitable agent found for task: {task.id}")
                task.status = TaskStatus.FAILED
                task.error = "No suitable agent available"
                return False
            
            # Execute task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task.id] = task
            
            logger.info(f"Executing task {task.id} with agent {agent.name}")
            
            # Call agent's execute method
            if hasattr(agent.instance, 'execute_action'):
                result = await agent.instance.execute_action(
                    task.action, task.parameters
                )
                task.result = result
                task.status = TaskStatus.COMPLETED
            else:
                raise AttributeError(f"Agent {agent.name} missing execute_action method")
            
            task.completed_at = datetime.now()
            agent.last_used = datetime.now()
            
            # Store in memory
            await self.memory_system.store_task_result(task)
            
            # Execute callback if provided
            if task.callback:
                await task.callback(task)
            
            logger.info(f"Task {task.id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            return False
        
        finally:
            # Move task from running to completed
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            self.completed_tasks.append(task)
    
    def _find_agent_for_task(self, task: Task) -> Optional[AgentInfo]:
        """Find the best agent for executing a task"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if (agent.agent_type == task.agent_type and 
                agent.is_active and
                task.action in agent.capabilities):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Return least recently used agent
        return min(suitable_agents, 
                  key=lambda a: a.last_used or datetime.min)
    
    def _check_task_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {t.id for t in self.completed_tasks 
                            if t.status == TaskStatus.COMPLETED}
        
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)
    
    async def process_task_queue(self):
        """Process tasks in the queue"""
        while self.task_queue and not self.emergency_stop:
            # Find next executable task
            executable_task = None
            for i, task in enumerate(self.task_queue):
                if self._check_task_dependencies(task):
                    executable_task = self.task_queue.pop(i)
                    break
            
            if not executable_task:
                # No executable tasks, wait a bit
                await asyncio.sleep(1)
                continue
            
            # Execute the task
            await self.execute_task(executable_task)
            
            # Small delay between tasks
            await asyncio.sleep(0.1)
    
    async def start(self):
        """Start the orchestrator main loop"""
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return
        
        self.is_running = True
        self.emergency_stop = False
        
        logger.info("Starting LazyCoder Agent Orchestrator")
        
        try:
            # Main orchestrator loop
            while self.is_running and not self.emergency_stop:
                # Process task queue
                await self.process_task_queue()
                
                # Check system health
                await self._health_check()
                
                # Small delay in main loop
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Orchestrator main loop error: {e}")
            await self.emergency_stop_system()
        
        finally:
            self.is_running = False
            logger.info("Orchestrator stopped")
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping orchestrator...")
        self.is_running = False
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
    
    async def emergency_stop_system(self):
        """Emergency stop - halt all operations immediately"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_stop = True
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.status = TaskStatus.CANCELLED
            task.error = "Emergency stop activated"
            task.completed_at = datetime.now()
        
        # Clear task queue
        for task in self.task_queue:
            task.status = TaskStatus.CANCELLED
            task.error = "Emergency stop activated"
            self.completed_tasks.append(task)
        
        self.task_queue.clear()
        self.running_tasks.clear()
    
    async def _health_check(self):
        """Perform system health checks"""
        try:
            # Check if safety monitor detects issues
            if self.safety_monitor and await self.safety_monitor.check_safety():
                logger.warning("Safety monitor detected issues")
                # Handle safety issues
            
            # Check agent health
            for agent in self.agents.values():
                if hasattr(agent.instance, 'health_check'):
                    if not await agent.instance.health_check():
                        logger.warning(f"Agent {agent.name} failed health check")
                        agent.is_active = False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "is_running": self.is_running,
            "emergency_stop": self.emergency_stop,
            "registered_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.is_active),
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "autonomy_level": (self.autonomy_controller.current_level.name 
                             if self.autonomy_controller else "unknown")
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
        else:
            # Check completed tasks
            task = next((t for t in self.completed_tasks if t.id == task_id), None)
            if not task:
                # Check queued tasks
                task = next((t for t in self.task_queue if t.id == task_id), None)
        
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error
        }