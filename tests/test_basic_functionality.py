"""
Basic functionality tests for LazyCoder
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from pathlib import Path

from src.core.config import LazyCoderConfig
from src.agents.orchestrator import AgentOrchestrator, TaskPriority
from src.agents.autonomy import AutonomyController, AutonomyLevel
from src.agents.file_processing import FileProcessingAgent, FileType


class TestConfiguration:
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test loading configuration from YAML"""
        # Create temporary config file
        config_content = """
app:
  name: "LazyCoder"
  version: "1.0.0"

ai_providers:
  openai:
    api_key: "test-key"
    model: "gpt-4-turbo-preview"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            # Test loading
            config = LazyCoderConfig(config_path)
            assert config.app_name == "LazyCoder"
            
            # Test AI provider config
            openai_config = config.get_ai_provider_config('openai')
            assert openai_config.model == "gpt-4-turbo-preview"
            
        finally:
            os.unlink(config_path)


class TestAutonomyController:
    """Test autonomy control system"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return LazyCoderConfig()
    
    @pytest.fixture
    def autonomy_controller(self, config):
        """Create autonomy controller"""
        return AutonomyController(config)
    
    def test_autonomy_level_setting(self, autonomy_controller):
        """Test setting autonomy levels"""
        # Test initial level
        assert autonomy_controller.current_level == AutonomyLevel.MANUAL
        
        # Test changing level
        result = autonomy_controller.set_autonomy_level(AutonomyLevel.SEMI_AUTO)
        assert result is True
        assert autonomy_controller.current_level == AutonomyLevel.SEMI_AUTO
    
    def test_emergency_stop(self, autonomy_controller):
        """Test emergency stop functionality"""
        # Activate emergency stop
        autonomy_controller.activate_emergency_stop()
        assert autonomy_controller.emergency_stop_active is True
        
        # Deactivate emergency stop
        autonomy_controller.deactivate_emergency_stop()
        assert autonomy_controller.emergency_stop_active is False
    
    @pytest.mark.asyncio
    async def test_action_classification(self, autonomy_controller):
        """Test action classification and permission checking"""
        await autonomy_controller.initialize()
        
        # Test safe action in manual mode
        can_execute = await autonomy_controller.can_execute_action(
            "file_read", {"path": "/safe/path/file.txt"}
        )
        # Should require confirmation in manual mode
        assert can_execute is False
        
        # Test with full auto mode
        autonomy_controller.set_autonomy_level(AutonomyLevel.FULL_AUTO)
        can_execute = await autonomy_controller.can_execute_action(
            "file_read", {"path": "/safe/path/file.txt"}
        )
        # Should allow safe actions in full auto
        assert can_execute is True


class TestFileProcessing:
    """Test file processing functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with mock API keys"""
        config = LazyCoderConfig()
        # Set mock API key for testing
        config.openai_api_key = "test-key"
        return config
    
    @pytest.fixture
    def file_agent(self, config):
        """Create file processing agent"""
        return FileProcessingAgent(config)
    
    def test_file_type_detection(self, file_agent):
        """Test file type detection"""
        # Create temporary test files
        test_files = {
            "test.py": FileType.CODE,
            "test.pdf": FileType.DOCUMENT,
            "test.jpg": FileType.IMAGE,
            "test.mp3": FileType.AUDIO,
            "test.mp4": FileType.VIDEO
        }
        
        for filename, expected_type in test_files.items():
            with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as f:
                f.write(b"test content")
                temp_path = f.name
            
            try:
                file_info = file_agent._analyze_file(temp_path)
                assert file_info.type == expected_type
            finally:
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_text_file_processing(self, file_agent):
        """Test processing of text files"""
        # Create temporary Python file
        python_code = '''
def hello_world():
    """Print hello world"""
    print("Hello, World!")
    return True

import os
import sys
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_path = f.name
        
        try:
            # Mock the OpenAI client to avoid API calls in tests
            file_agent.openai_client = None
            
            # Process file
            result = await file_agent.process_file(temp_path)
            
            assert result.status == "completed"
            assert result.file_info.type == FileType.CODE
            assert "hello_world" in result.content_analysis.text_content
            assert len(result.content_analysis.functions) > 0
            assert len(result.content_analysis.imports) > 0
            
        finally:
            os.unlink(temp_path)


class TestOrchestrator:
    """Test agent orchestrator"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = LazyCoderConfig()
        config.openai_api_key = "test-key"
        return config
    
    @pytest_asyncio.fixture
    async def orchestrator(self, config):
        """Create and initialize orchestrator"""
        orchestrator = AgentOrchestrator(config)
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator, config):
        """Test agent registration"""
        # Create mock agent
        class MockAgent:
            async def execute_action(self, action, parameters):
                return {"result": "success"}
            
            async def health_check(self):
                return True
            
            def get_capabilities(self):
                return ["test_action"]
        
        mock_agent = MockAgent()
        
        # Register agent
        agent_name = orchestrator.register_agent(
            mock_agent, "test_agent", ["test_action"]
        )
        
        assert agent_name in orchestrator.agents
        assert orchestrator.agents[agent_name].agent_type == "test_agent"
        assert "test_action" in orchestrator.agents[agent_name].capabilities
    
    @pytest.mark.asyncio
    async def test_task_creation_and_execution(self, orchestrator, config):
        """Test task creation and execution"""
        # Create mock agent
        class MockAgent:
            async def execute_action(self, action, parameters):
                return {"result": "success", "parameters": parameters}
            
            async def health_check(self):
                return True
            
            def get_capabilities(self):
                return ["test_action"]
        
        mock_agent = MockAgent()
        
        # Register agent
        orchestrator.register_agent(mock_agent, "test_agent", ["test_action"])
        
        # Create task
        task_id = orchestrator.create_task(
            name="Test Task",
            description="Test task description",
            agent_type="test_agent",
            action="test_action",
            parameters={"test_param": "test_value"},
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert len(orchestrator.task_queue) == 1
        
        # Process task queue
        await orchestrator.process_task_queue()
        
        # Check task completion
        assert len(orchestrator.completed_tasks) == 1
        completed_task = orchestrator.completed_tasks[0]
        assert completed_task.id == task_id
        assert completed_task.result["result"] == "success"
    
    def test_orchestrator_status(self, config):
        """Test orchestrator status reporting"""
        orchestrator = AgentOrchestrator(config)
        status = orchestrator.get_status()
        
        assert "is_running" in status
        assert "emergency_stop" in status
        assert "registered_agents" in status
        assert status["registered_agents"] == 0


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_file_processing_workflow(self):
        """Test complete file processing workflow"""
        # Create test configuration
        config = LazyCoderConfig()
        config.openai_api_key = "test-key"
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(config)
        await orchestrator.initialize()
        
        # Create and register file processing agent
        file_agent = FileProcessingAgent(config)
        file_agent.openai_client = None  # Mock to avoid API calls
        
        orchestrator.register_agent(
            file_agent, "file_processing", file_agent.get_capabilities()
        )
        
        # Create test file
        test_content = "print('Hello from LazyCoder!')"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Create processing task
            task_id = orchestrator.create_task(
                name="Process Python File",
                description="Process test Python file",
                agent_type="file_processing",
                action="process_file",
                parameters={"file_path": temp_path, "context": "test"}
            )
            
            # Process task
            await orchestrator.process_task_queue()
            
            # Verify results
            assert len(orchestrator.completed_tasks) == 1
            task = orchestrator.completed_tasks[0]
            assert task.id == task_id
            assert task.result.status == "completed"
            assert task.result.file_info.type == FileType.CODE
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])