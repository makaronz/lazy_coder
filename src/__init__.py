"""
LazyCoder - Autonomous AI Agent for Cursor IDE
Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "LazyCoder Team"
__email__ = "contact@lazycoder.dev"
__description__ = "Autonomous AI Agent for Cursor IDE with God Mode capabilities"

from .core.config import LazyCoderConfig, load_config, get_config
from .agents.orchestrator import AgentOrchestrator, Task, TaskStatus, TaskPriority
from .agents.autonomy import AutonomyController, AutonomyLevel
from .agents.file_processing import FileProcessingAgent, FileType, ProcessingResult

__all__ = [
    # Core
    "LazyCoderConfig",
    "load_config",
    "get_config",
    
    # Orchestrator
    "AgentOrchestrator", 
    "Task",
    "TaskStatus",
    "TaskPriority",
    
    # Autonomy
    "AutonomyController",
    "AutonomyLevel",
    
    # File Processing
    "FileProcessingAgent",
    "FileType", 
    "ProcessingResult",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]