"""
LazyCoder - Autonomous AI Agent for Cursor IDE
Main application entry point and CLI interface.
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

try:
    # Try relative imports first (when run as module)
    from .core.config import load_config, get_config
    from .agents.orchestrator import AgentOrchestrator
    from .agents.autonomy import AutonomyController, AutonomyLevel
    from .agents.file_processing import FileProcessingAgent
except ImportError:
    # Fall back to absolute imports (when run as script)
    from core.config import load_config, get_config
    from agents.orchestrator import AgentOrchestrator
    from agents.autonomy import AutonomyController, AutonomyLevel
    from agents.file_processing import FileProcessingAgent


# CLI app
app = typer.Typer(
    name="lazycoder",
    help="LazyCoder - Autonomous AI Agent for Cursor IDE with God Mode capabilities",
    add_completion=False
)

console = Console()


class LazyCoderApp:
    """Main LazyCoder application class"""
    
    def __init__(self):
        self.config = None
        self.orchestrator = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self, config_path: str = "config/settings.yaml"):
        """Initialize the LazyCoder application"""
        try:
            # Load configuration
            self.config = load_config(config_path)
            
            # Setup logging
            self._setup_logging()
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(self.config)
            await self.orchestrator.initialize()
            
            # Register agents
            await self._register_agents()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            logger.info("LazyCoder application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LazyCoder: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Add file handler
        log_file = Path("logs/lazycoder.log")
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="1 day",
            retention="30 days"
        )
    
    async def _register_agents(self):
        """Register all agents with the orchestrator"""
        try:
            # File Processing Agent
            file_agent = FileProcessingAgent(self.config)
            await file_agent.initialize()
            self.orchestrator.register_agent(
                file_agent, 
                "file_processing", 
                file_agent.get_capabilities()
            )
            
            # TODO: Register other agents (GitHub, Cursor, etc.)
            
            logger.info("All agents registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register agents: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start the LazyCoder application"""
        if self.is_running:
            logger.warning("LazyCoder is already running")
            return
        
        self.is_running = True
        
        console.print(Panel.fit(
            "[bold green]🚀 LazyCoder Started[/bold green]\n"
            "[cyan]Autonomous AI Agent for Cursor IDE[/cyan]\n"
            f"[yellow]Autonomy Level: {self.orchestrator.autonomy_controller.current_level.value}[/yellow]",
            title="LazyCoder",
            border_style="green"
        ))
        
        try:
            # Start orchestrator
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Graceful shutdown
            await self.shutdown()
            
            # Wait for orchestrator to finish
            await orchestrator_task
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            await self.emergency_shutdown()
        
        finally:
            self.is_running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        console.print(Panel.fit(
            "[bold red]🛑 LazyCoder Stopped[/bold red]\n"
            "[yellow]All operations completed safely[/yellow]",
            title="Shutdown",
            border_style="red"
        ))
    
    async def emergency_shutdown(self):
        """Emergency shutdown"""
        logger.critical("Initiating emergency shutdown...")
        
        if self.orchestrator:
            await self.orchestrator.emergency_stop_system()
        
        console.print(Panel.fit(
            "[bold red]🚨 EMERGENCY STOP[/bold red]\n"
            "[red]All operations halted immediately[/red]",
            title="Emergency Shutdown",
            border_style="red"
        ))


# Global app instance
lazycoder_app = LazyCoderApp()


@app.command()
def start(
    config: Optional[str] = typer.Option(
        "config/settings.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    autonomy: Optional[str] = typer.Option(
        None,
        "--autonomy", "-a", 
        help="Set autonomy level (manual, semi_auto, full_auto, dry_run)"
    )
):
    """Start LazyCoder autonomous agent"""
    
    async def _start():
        try:
            await lazycoder_app.initialize(config)
            
            # Set autonomy level if specified
            if autonomy:
                try:
                    level = AutonomyLevel(autonomy)
                    lazycoder_app.orchestrator.autonomy_controller.set_autonomy_level(level)
                    console.print(f"[green]Autonomy level set to: {level.value}[/green]")
                except ValueError:
                    console.print(f"[red]Invalid autonomy level: {autonomy}[/red]")
                    return
            
            await lazycoder_app.start()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutdown requested by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Application failed: {e}")
    
    asyncio.run(_start())


@app.command()
def status():
    """Show LazyCoder status"""
    # This would connect to running instance to get status
    console.print("[yellow]Status command not yet implemented[/yellow]")


@app.command()
def process_file(
    file_path: str = typer.Argument(..., help="Path to file to process"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Processing context")
):
    """Process a single file with AI analysis"""
    
    async def _process():
        try:
            await lazycoder_app.initialize()
            
            # Get file processing agent
            file_agent = None
            for agent_info in lazycoder_app.orchestrator.agents.values():
                if agent_info.agent_type == "file_processing":
                    file_agent = agent_info.instance
                    break
            
            if not file_agent:
                console.print("[red]File processing agent not available[/red]")
                return
            
            console.print(f"[cyan]Processing file: {file_path}[/cyan]")
            
            # Process file
            result = await file_agent.process_file(file_path, context or "")
            
            # Display results
            table = Table(title="File Processing Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("File ID", result.file_id)
            table.add_row("File Type", result.file_info.type.value)
            table.add_row("Size", f"{result.file_info.size} bytes")
            table.add_row("Processing Time", f"{result.processing_time:.2f}s")
            table.add_row("Status", result.status)
            
            if result.content_analysis.summary:
                table.add_row("Summary", result.content_analysis.summary[:100] + "...")
            
            if result.content_analysis.insights:
                table.add_row("Insights", "\n".join(result.content_analysis.insights))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error processing file: {e}[/red]")
    
    asyncio.run(_process())


@app.command()
def set_autonomy(
    level: str = typer.Argument(..., help="Autonomy level (manual, semi_auto, full_auto, dry_run)")
):
    """Set autonomy level"""
    try:
        autonomy_level = AutonomyLevel(level)
        console.print(f"[green]Autonomy level would be set to: {autonomy_level.value}[/green]")
        console.print("[yellow]Note: This requires a running LazyCoder instance[/yellow]")
    except ValueError:
        console.print(f"[red]Invalid autonomy level: {level}[/red]")
        console.print("[cyan]Valid levels: manual, semi_auto, full_auto, dry_run[/cyan]")


@app.command()
def emergency_stop():
    """Activate emergency stop"""
    console.print("[red]🚨 EMERGENCY STOP ACTIVATED[/red]")
    console.print("[yellow]Note: This requires a running LazyCoder instance[/yellow]")


@app.command()
def version():
    """Show LazyCoder version"""
    console.print(Panel.fit(
        "[bold cyan]LazyCoder v1.0.0[/bold cyan]\n"
        "[green]Autonomous AI Agent for Cursor IDE[/green]\n"
        "[yellow]Built with ❤️ for developers[/yellow]",
        title="Version Info",
        border_style="cyan"
    ))


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()