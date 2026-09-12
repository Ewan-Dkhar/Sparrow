"""Rich terminal renderer for Sparrow.

Provides styled banners, hardware status cards, Markdown live-streaming,
and formatted panels for user and assistant messages.
"""

from typing import Optional, Tuple
import torch
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.model.sparrow_moe import SparrowConfig


class TerminalRenderer:
    """Handles formatted Rich terminal presentation for the Sparrow CLI."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_banner(self, config: SparrowConfig) -> None:
        """Displays stylized ASCII banner and model/hardware specifications."""
        banner_text = Text(
            r"""
   ____                                   
  / __/ ___  ___ _  ____  ____  ___  __    __
 _\ \  / _ \/ _ `/ / __/ / __/ / _ \ | |/|/ /
/___/ / .__/\_,_/ /_/   /_/    \___/ |__,__/ 
     /_/                                      
       Sparse Mixture of Experts (MoE) Mini-LLM
""",
            style="bold cyan",
        )

        total_p, active_p = config.count_parameters()

        # Hardware info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hw_str = f"[green]{device_name} ({total_vram_gb:.1f} GB VRAM)[/green]"
        else:
            hw_str = "[yellow]CPU Mode (CUDA not detected)[/yellow]"

        table = Table(box=ROUNDED, border_style="bright_blue", show_header=False, expand=True)
        table.add_column("Spec", style="bold white", width=22)
        table.add_column("Value", style="cyan")

        table.add_row("Architecture", f"{config.name} (Decoder-Only)")
        table.add_row("Total Parameters", f"{total_p / 1e6:.1f}M")
        table.add_row("Active Parameters", f"[bold green]{active_p / 1e6:.1f}M[/bold green] (Top-{config.top_k} of {config.n_experts} experts)")
        table.add_row("Layers / Heads", f"{config.n_layers} layers | {config.n_heads} Q-heads | {config.n_kv_heads} KV-heads (GQA)")
        table.add_row("Hidden / FFN Dim", f"d_model={config.d_model} | d_ff={config.d_ff} (SwiGLU)")
        table.add_row("Context Window", f"{config.max_seq_len} tokens (RoPE)")
        table.add_row("Hardware Target", hw_str)

        panel = Panel(
            table,
            title="[bold magenta] Sparrow MoE Engine [/bold magenta]",
            subtitle="[dim]Type /help for commands | Ctrl+D to exit[/dim]",
            border_style="magenta",
            box=ROUNDED,
        )

        self.console.print(banner_text)
        self.console.print(panel)
        self.console.print()

    def print_user_message(self, text: str) -> None:
        """Displays user input inside a styled panel."""
        panel = Panel(
            Text(text, style="white"),
            title="[bold yellow]You[/bold yellow]",
            title_align="left",
            border_style="yellow",
            box=ROUNDED,
        )
        self.console.print(panel)

    def print_system_message(self, text: str) -> None:
        """Displays a system notification."""
        self.console.print(f"[bold blue]ℹ System:[/bold blue] {text}")

    def print_error(self, text: str) -> None:
        """Displays an error notification."""
        self.console.print(f"[bold red]✖ Error:[/bold red] {text}")

    def live_stream_markdown(self) -> Tuple[Live, callable]:
        """Creates a Rich Live context for streaming assistant Markdown responses.

        Returns:
            (live, update_fn) tuple where update_fn(text_delta) streams updates.
        """
        accumulated_text = [""]

        live = Live(
            Panel(
                Markdown(""),
                title="[bold cyan]Sparrow[/bold cyan]",
                title_align="left",
                border_style="cyan",
                box=ROUNDED,
            ),
            console=self.console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )

        def update(token_text: str) -> None:
            accumulated_text[0] += token_text
            live.update(
                Panel(
                    Markdown(accumulated_text[0]),
                    title="[bold cyan]Sparrow[/bold cyan]",
                    title_align="left",
                    border_style="cyan",
                    box=ROUNDED,
                )
            )

        return live, update

    def print_help(self) -> None:
        """Displays slash command options in a clean table."""
        table = Table(box=ROUNDED, border_style="cyan", title="Available Commands")
        table.add_column("Command", style="bold green")
        table.add_column("Description", style="white")

        table.add_row("/help", "Show this commands reference")
        table.add_row("/clear", "Reset chat history and free KV cache context")
        table.add_row("/gpu", "Display GPU VRAM utilization and device status")
        table.add_row("/params", "Print detailed parameter breakdown")
        table.add_row("/temp <val>", "Set sampling temperature (e.g., /temp 0.8)")
        table.add_row("/exit", "Exit the Sparrow CLI")

        self.console.print(table)
