"""
LazyCoder Autonomy Controller
Manages different levels of autonomous operation and user confirmation requirements.
"""

import os
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from loguru import logger

from ...core.config import LazyCoderConfig, AutonomyConfig


class AutonomyLevel(Enum):
    """Autonomy operation levels"""
    MANUAL = "manual"           # User confirms each action
    SEMI_AUTO = "semi_auto"     # Auto-execute safe actions, confirm risky ones
    FULL_AUTO = "full_auto"     # Full autonomous operation with safety checks
    DRY_RUN = "dry_run"         # Simulate actions without execution


@dataclass
class ActionClassification:
    """Classification of an action for autonomy decisions"""
    action_type: str
    risk_level: str  # low, medium, high, critical
    requires_confirmation: bool
    safety_checks: List[str]
    rollback_possible: bool


class AutonomyController:
    """
    Controls the level of autonomous operation and manages user confirmation workflows.
    Implements safety checks and emergency stop functionality.
    """
    
    def __init__(self, config: LazyCoderConfig):
        self.config = config
        self.autonomy_config = config.get_autonomy_config()
        self.current_level = AutonomyLevel(self.autonomy_config.default_level)
        self.emergency_stop_active = False
        self.action_history: List[Dict[str, Any]] = []
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}
        
        # Action classification rules
        self.action_classifications = self._initialize_action_classifications()
        
        # User confirmation callback
        self.confirmation_callback: Optional[Callable] = None
        
        logger.info(f"AutonomyController initialized with level: {self.current_level.value}")
    
    async def initialize(self):
        """Initialize the autonomy controller"""
        try:
            # Load any saved state or preferences
            await self._load_user_preferences()
            
            # Initialize safety monitoring
            await self._initialize_safety_monitoring()
            
            logger.info("AutonomyController initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutonomyController: {e}")
            raise
    
    def _initialize_action_classifications(self) -> Dict[str, ActionClassification]:
        """Initialize action classification rules"""
        return {
            # File operations
            "file_read": ActionClassification(
                action_type="file_operation",
                risk_level="low",
                requires_confirmation=False,
                safety_checks=["path_validation"],
                rollback_possible=True
            ),
            "file_write": ActionClassification(
                action_type="file_operation", 
                risk_level="medium",
                requires_confirmation=True,
                safety_checks=["path_validation", "backup_creation"],
                rollback_possible=True
            ),
            "file_delete": ActionClassification(
                action_type="file_operation",
                risk_level="high",
                requires_confirmation=True,
                safety_checks=["path_validation", "backup_creation", "whitelist_check"],
                rollback_possible=False
            ),
            
            # Git operations
            "git_commit": ActionClassification(
                action_type="git_operation",
                risk_level="medium",
                requires_confirmation=True,
                safety_checks=["branch_check", "diff_review"],
                rollback_possible=True
            ),
            "git_push": ActionClassification(
                action_type="git_operation",
                risk_level="high",
                requires_confirmation=True,
                safety_checks=["branch_check", "remote_validation"],
                rollback_possible=False
            ),
            
            # System operations
            "system_command": ActionClassification(
                action_type="system_operation",
                risk_level="critical",
                requires_confirmation=True,
                safety_checks=["command_whitelist", "privilege_check"],
                rollback_possible=False
            ),
            
            # Cursor IDE operations
            "cursor_open_file": ActionClassification(
                action_type="cursor_operation",
                risk_level="low",
                requires_confirmation=False,
                safety_checks=["file_exists"],
                rollback_possible=True
            ),
            "cursor_edit_code": ActionClassification(
                action_type="cursor_operation",
                risk_level="medium",
                requires_confirmation=True,
                safety_checks=["syntax_validation", "backup_creation"],
                rollback_possible=True
            ),
            
            # AI operations
            "ai_code_generation": ActionClassification(
                action_type="ai_operation",
                risk_level="medium",
                requires_confirmation=True,
                safety_checks=["code_review", "security_scan"],
                rollback_possible=True
            ),
            "ai_file_analysis": ActionClassification(
                action_type="ai_operation",
                risk_level="low",
                requires_confirmation=False,
                safety_checks=["content_validation"],
                rollback_possible=True
            )
        }
    
    async def can_execute_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """
        Determine if an action can be executed based on current autonomy level.
        Returns True if action can proceed, False if user confirmation needed.
        """
        if self.emergency_stop_active:
            logger.warning("Emergency stop active - blocking all actions")
            return False
        
        # Get action classification
        classification = self.action_classifications.get(action)
        if not classification:
            logger.warning(f"Unknown action: {action} - requiring confirmation")
            return await self._request_user_confirmation(action, parameters)
        
        # Check based on autonomy level
        if self.current_level == AutonomyLevel.DRY_RUN:
            logger.info(f"DRY RUN: Would execute {action} with {parameters}")
            return False  # Don't actually execute in dry run
        
        elif self.current_level == AutonomyLevel.MANUAL:
            # Always require confirmation in manual mode
            return await self._request_user_confirmation(action, parameters, classification)
        
        elif self.current_level == AutonomyLevel.SEMI_AUTO:
            # Check if this action type requires confirmation
            if action in self.autonomy_config.require_confirmation_for:
                return await self._request_user_confirmation(action, parameters, classification)
            
            # Auto-execute if risk level is low
            if classification.risk_level in ["low"]:
                return await self._perform_safety_checks(action, parameters, classification)
            else:
                return await self._request_user_confirmation(action, parameters, classification)
        
        elif self.current_level == AutonomyLevel.FULL_AUTO:
            # Perform safety checks and execute if passed
            return await self._perform_safety_checks(action, parameters, classification)
        
        return False
    
    async def _perform_safety_checks(self, action: str, parameters: Dict[str, Any], 
                                   classification: ActionClassification) -> bool:
        """Perform safety checks for an action"""
        try:
            for check in classification.safety_checks:
                if not await self._execute_safety_check(check, action, parameters):
                    logger.warning(f"Safety check failed: {check} for action {action}")
                    return False
            
            logger.info(f"All safety checks passed for action: {action}")
            return True
            
        except Exception as e:
            logger.error(f"Safety check error for action {action}: {e}")
            return False
    
    async def _execute_safety_check(self, check_name: str, action: str, 
                                  parameters: Dict[str, Any]) -> bool:
        """Execute a specific safety check"""
        try:
            if check_name == "path_validation":
                return await self._validate_file_path(parameters.get("path", ""))
            
            elif check_name == "backup_creation":
                return await self._create_backup(parameters.get("path", ""))
            
            elif check_name == "whitelist_check":
                return await self._check_whitelist(parameters)
            
            elif check_name == "command_whitelist":
                return await self._check_command_whitelist(parameters.get("command", ""))
            
            elif check_name == "branch_check":
                return await self._validate_git_branch(parameters)
            
            elif check_name == "syntax_validation":
                return await self._validate_syntax(parameters)
            
            elif check_name == "security_scan":
                return await self._security_scan(parameters)
            
            else:
                logger.warning(f"Unknown safety check: {check_name}")
                return True  # Default to allowing unknown checks
                
        except Exception as e:
            logger.error(f"Safety check {check_name} failed: {e}")
            return False
    
    async def _validate_file_path(self, path: str) -> bool:
        """Validate file path is safe"""
        if not path:
            return False
        
        # Check against blacklisted paths
        security_config = self.config.get_security_config()
        for blacklisted_path in security_config.blacklist_paths:
            if path.startswith(blacklisted_path):
                logger.warning(f"Path {path} is blacklisted")
                return False
        
        return True
    
    async def _create_backup(self, path: str) -> bool:
        """Create backup of file before modification"""
        try:
            if not path or not os.path.exists(path):
                return True  # No backup needed for non-existent files
            
            backup_path = f"{path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup for {path}: {e}")
            return False
    
    async def _check_whitelist(self, parameters: Dict[str, Any]) -> bool:
        """Check if operation is whitelisted"""
        # Implement whitelist checking logic
        return True
    
    async def _check_command_whitelist(self, command: str) -> bool:
        """Check if system command is whitelisted"""
        security_config = self.config.get_security_config()
        
        for blacklisted_cmd in security_config.blacklist_commands:
            if blacklisted_cmd in command:
                logger.warning(f"Command contains blacklisted term: {blacklisted_cmd}")
                return False
        
        return True
    
    async def _validate_git_branch(self, parameters: Dict[str, Any]) -> bool:
        """Validate git branch operations"""
        # Implement git branch validation
        return True
    
    async def _validate_syntax(self, parameters: Dict[str, Any]) -> bool:
        """Validate code syntax"""
        # Implement syntax validation
        return True
    
    async def _security_scan(self, parameters: Dict[str, Any]) -> bool:
        """Perform security scan on code"""
        # Implement security scanning
        return True
    
    async def _request_user_confirmation(self, action: str, parameters: Dict[str, Any],
                                       classification: Optional[ActionClassification] = None) -> bool:
        """Request user confirmation for an action"""
        confirmation_id = f"confirm_{datetime.now().timestamp()}"
        
        confirmation_data = {
            "id": confirmation_id,
            "action": action,
            "parameters": parameters,
            "classification": classification,
            "timestamp": datetime.now(),
            "status": "pending"
        }
        
        self.pending_confirmations[confirmation_id] = confirmation_data
        
        if self.confirmation_callback:
            try:
                result = await self.confirmation_callback(confirmation_data)
                confirmation_data["status"] = "confirmed" if result else "denied"
                return result
            except Exception as e:
                logger.error(f"Confirmation callback failed: {e}")
                confirmation_data["status"] = "error"
                return False
        else:
            logger.warning("No confirmation callback set - denying action")
            confirmation_data["status"] = "denied"
            return False
    
    def set_autonomy_level(self, level: AutonomyLevel) -> bool:
        """Change the autonomy level"""
        try:
            old_level = self.current_level
            self.current_level = level
            
            logger.info(f"Autonomy level changed from {old_level.value} to {level.value}")
            
            # Log the change
            self.action_history.append({
                "timestamp": datetime.now(),
                "action": "autonomy_level_change",
                "old_level": old_level.value,
                "new_level": level.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to change autonomy level: {e}")
            return False
    
    def activate_emergency_stop(self) -> None:
        """Activate emergency stop - halt all autonomous operations"""
        self.emergency_stop_active = True
        logger.critical("EMERGENCY STOP ACTIVATED - All autonomous operations halted")
        
        # Cancel all pending confirmations
        for confirmation in self.pending_confirmations.values():
            confirmation["status"] = "cancelled_emergency_stop"
    
    def deactivate_emergency_stop(self) -> None:
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        logger.info("Emergency stop deactivated - Autonomous operations can resume")
    
    def set_confirmation_callback(self, callback: Callable) -> None:
        """Set callback function for user confirmations"""
        self.confirmation_callback = callback
        logger.info("User confirmation callback set")
    
    async def _load_user_preferences(self):
        """Load user preferences and settings"""
        # Implement loading of user preferences
        pass
    
    async def _initialize_safety_monitoring(self):
        """Initialize safety monitoring systems"""
        # Implement safety monitoring initialization
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autonomy controller status"""
        return {
            "current_level": self.current_level.value,
            "emergency_stop_active": self.emergency_stop_active,
            "pending_confirmations": len(self.pending_confirmations),
            "action_history_count": len(self.action_history),
            "available_levels": [level.value for level in AutonomyLevel]
        }
    
    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """Get list of pending user confirmations"""
        return [
            {
                "id": conf["id"],
                "action": conf["action"],
                "timestamp": conf["timestamp"].isoformat(),
                "status": conf["status"]
            }
            for conf in self.pending_confirmations.values()
        ]
    
    def respond_to_confirmation(self, confirmation_id: str, approved: bool) -> bool:
        """Respond to a pending confirmation"""
        if confirmation_id not in self.pending_confirmations:
            logger.warning(f"Unknown confirmation ID: {confirmation_id}")
            return False
        
        confirmation = self.pending_confirmations[confirmation_id]
        confirmation["status"] = "confirmed" if approved else "denied"
        confirmation["response_time"] = datetime.now()
        
        logger.info(f"Confirmation {confirmation_id} {'approved' if approved else 'denied'}")
        
        # Remove from pending
        del self.pending_confirmations[confirmation_id]
        
        return True