"""
LazyCoder Configuration Management
Handles loading and validation of configuration from YAML and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIProviderConfig(BaseSettings):
    """Configuration for AI API providers"""
    api_key: str = Field(..., description="API key for the provider")
    model: str = Field(..., description="Model name to use")
    max_tokens: int = Field(4096, description="Maximum tokens per request")
    temperature: float = Field(0.7, description="Temperature for generation")


class AutonomyConfig(BaseSettings):
    """Configuration for autonomy control system"""
    default_level: str = Field("manual", description="Default autonomy level")
    emergency_stop_enabled: bool = Field(True, description="Enable emergency stop")
    rollback_enabled: bool = Field(True, description="Enable rollback functionality")
    max_autonomous_actions: int = Field(50, description="Max actions in autonomous mode")


class GitHubConfig(BaseSettings):
    """Configuration for GitHub integration"""
    token: Optional[str] = Field(None, description="GitHub personal access token")
    default_branch: str = Field("main", description="Default branch name")
    auto_commit: bool = Field(False, description="Enable auto-commit")
    commit_message_template: str = Field(
        "[LazyCoder] {action}: {description}",
        description="Template for commit messages"
    )


class CursorConfig(BaseSettings):
    """Configuration for Cursor IDE integration"""
    executable_path: str = Field(
        "/Applications/Cursor.app/Contents/MacOS/Cursor",
        description="Path to Cursor executable"
    )
    workspace_path: Optional[str] = Field(None, description="Cursor workspace path")
    chat_integration: bool = Field(True, description="Enable chat integration")
    gui_automation: bool = Field(True, description="Enable GUI automation")


class FileProcessingConfig(BaseSettings):
    """Configuration for file processing system"""
    max_file_size: str = Field("100MB", description="Maximum file size")
    local_path: str = Field("./uploads", description="Local storage path")
    vector_db_collection: str = Field("file_contents", description="Vector DB collection")
    retention_days: int = Field(30, description="File retention period")
    use_openai_vision: bool = Field(True, description="Use OpenAI Vision for images")
    use_whisper: bool = Field(True, description="Use Whisper for audio")
    whisper_model: str = Field("base", description="Whisper model size")


class MemoryConfig(BaseSettings):
    """Configuration for memory system"""
    provider: str = Field("chromadb", description="Vector database provider")
    collection_name: str = Field("lazycoder_memory", description="Collection name")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model")
    max_context_length: int = Field(8192, description="Maximum context length")


class SecurityConfig(BaseSettings):
    """Configuration for security and safety"""
    whitelist_domains: list = Field(
        default_factory=lambda: ["github.com", "api.openai.com", "api.anthropic.com"]
    )
    blacklist_commands: list = Field(
        default_factory=lambda: ["rm -rf", "sudo", "chmod 777"]
    )
    blacklist_paths: list = Field(
        default_factory=lambda: ["/etc", "/usr/bin", "/System", "/private"]
    )
    emergency_keywords: list = Field(
        default_factory=lambda: ["STOP", "EMERGENCY", "ABORT"]
    )
    max_errors: int = Field(5, description="Maximum errors before emergency stop")


class LazyCoderConfig(BaseSettings):
    """Main configuration class for LazyCoder"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # App info
    app_name: str = Field("LazyCoder", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")
    
    # Paths
    cursor_workspace_path: Optional[str] = Field(None, env="CURSOR_WORKSPACE_PATH")
    
    # Development
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    test_mode: bool = Field(False, env="TEST_MODE")
    mock_apis: bool = Field(False, env="MOCK_APIS")
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with optional YAML file"""
        super().__init__()
        
        if config_path:
            self.load_yaml_config(config_path)
    
    def load_yaml_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Store YAML config for component-specific configurations
        self._yaml_config = yaml_config
    
    def get_ai_provider_config(self, provider: str) -> AIProviderConfig:
        """Get configuration for specific AI provider"""
        if hasattr(self, '_yaml_config'):
            provider_config = self._yaml_config.get('ai_providers', {}).get(provider, {})
            
            # Replace environment variables in API key
            api_key = provider_config.get('api_key', '')
            if api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var, '')
            
            return AIProviderConfig(
                api_key=api_key,
                model=provider_config.get('model', 'gpt-4-turbo-preview'),
                max_tokens=provider_config.get('max_tokens', 4096),
                temperature=provider_config.get('temperature', 0.7)
            )
        
        # Fallback to environment variables
        if provider == 'openai':
            return AIProviderConfig(
                api_key=self.openai_api_key or '',
                model='gpt-4-turbo-preview'
            )
        elif provider == 'anthropic':
            return AIProviderConfig(
                api_key=self.anthropic_api_key or '',
                model='claude-3-sonnet-20240229'
            )
        else:
            raise ValueError(f"Unknown AI provider: {provider}")
    
    def get_autonomy_config(self) -> AutonomyConfig:
        """Get autonomy configuration"""
        if hasattr(self, '_yaml_config'):
            autonomy_config = self._yaml_config.get('autonomy', {})
            return AutonomyConfig(**autonomy_config)
        return AutonomyConfig()
    
    def get_github_config(self) -> GitHubConfig:
        """Get GitHub configuration"""
        if hasattr(self, '_yaml_config'):
            github_config = self._yaml_config.get('github', {})
            # Replace environment variable in token
            token = github_config.get('token', '')
            if token.startswith('${') and token.endswith('}'):
                env_var = token[2:-1]
                token = os.getenv(env_var, '')
            
            return GitHubConfig(
                token=token,
                default_branch=github_config.get('default_branch', 'main'),
                auto_commit=github_config.get('auto_commit', False),
                commit_message_template=github_config.get(
                    'commit_message_template', 
                    '[LazyCoder] {action}: {description}'
                )
            )
        return GitHubConfig(token=self.github_token)
    
    def get_cursor_config(self) -> CursorConfig:
        """Get Cursor IDE configuration"""
        if hasattr(self, '_yaml_config'):
            cursor_config = self._yaml_config.get('cursor', {})
            return CursorConfig(**cursor_config)
        return CursorConfig(workspace_path=self.cursor_workspace_path)
    
    def get_file_processing_config(self) -> FileProcessingConfig:
        """Get file processing configuration"""
        if hasattr(self, '_yaml_config'):
            fp_config = self._yaml_config.get('file_processing', {})
            return FileProcessingConfig(**fp_config)
        return FileProcessingConfig()
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory system configuration"""
        if hasattr(self, '_yaml_config'):
            memory_config = self._yaml_config.get('memory', {}).get('vector_db', {})
            return MemoryConfig(**memory_config)
        return MemoryConfig()
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        if hasattr(self, '_yaml_config'):
            security_config = self._yaml_config.get('security', {})
            return SecurityConfig(**security_config)
        return SecurityConfig()


# Global configuration instance
config: Optional[LazyCoderConfig] = None


def load_config(config_path: str = "config/settings.yaml") -> LazyCoderConfig:
    """Load global configuration"""
    global config
    config = LazyCoderConfig(config_path)
    return config


def get_config() -> LazyCoderConfig:
    """Get global configuration instance"""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return config