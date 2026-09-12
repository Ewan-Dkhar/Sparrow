"""Read-Eval-Print Loop (REPL) for Sparrow combining prompt_toolkit and Rich.

Handles interactive user input with history, keyboard shortcuts, slash commands,
and connects token streams to the live Rich Markdown renderer.
"""

import sys
import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from src.cli.renderer import TerminalRenderer
from src.cli.session import ChatSession
from src.engine.generator import SparrowGenerator
from src.model.sparrow_moe import SparrowConfig, SparrowMoE


class SparrowREPL:
    """Interactive CLI REPL for chatting with the Sparrow MoE model.

    Coordinates prompt_toolkit for command line capture and Rich for live
    rendering of model completions.
    """

    def __init__(
        self,
        model: SparrowMoE,
        generator: SparrowGenerator,
        session: ChatSession,
        renderer: TerminalRenderer,
        config: SparrowConfig,
    ):
        self.model = model
        self.generator = generator
        self.session = session
        self.renderer = renderer
        self.config = config

        # Generation parameters (can be modified via slash commands)
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.9

        # prompt_toolkit prompt styling
        self.prompt_style = Style.from_dict({
            "prompt-bracket": "#00afff bold",
            "prompt-name": "#ff007f bold",
            "prompt-arrow": "#00ffaf bold",
        })

        self.prompt_session = PromptSession(
            history=InMemoryHistory(),
        )

    def handle_command(self, cmd_text: str) -> bool:
        """Processes slash commands.

        Args:
            cmd_text: Raw input string starting with '/'.

        Returns:
            True if input was handled as a command, False otherwise.
        """
        parts = cmd_text.strip().split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self.renderer.print_help()
            return True
        elif cmd == "/clear":
            self.session.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.renderer.print_system_message("Chat history and KV cache cleared.")
            return True
        elif cmd == "/gpu":
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                self.renderer.print_system_message(
                    f"VRAM Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB | Peak: {max_alloc:.1f}MB"
                )
            else:
                self.renderer.print_system_message("CUDA is not active. Running on CPU.")
            return True
        elif cmd == "/params":
            total_p, active_p = self.config.count_parameters()
            self.renderer.print_system_message(
                f"Total Parameters: {total_p:,} (~{total_p / 1e6:.1f}M) | "
                f"Active per Token: {active_p:,} (~{active_p / 1e6:.1f}M)"
            )
            return True
        elif cmd == "/temp":
            if len(parts) > 1:
                try:
                    new_temp = float(parts[1])
                    self.temperature = max(0.0, new_temp)
                    self.renderer.print_system_message(f"Temperature set to {self.temperature}")
                except ValueError:
                    self.renderer.print_error("Invalid temperature value. Must be a float.")
            else:
                self.renderer.print_system_message(f"Current temperature: {self.temperature}")
            return True
        elif cmd in ("/exit", "/quit"):
            self.renderer.print_system_message("Exiting Sparrow. Goodbye!")
            sys.exit(0)
        else:
            self.renderer.print_error(f"Unknown command '{cmd}'. Type /help for available commands.")
            return True

    def run(self) -> None:
        """Starts the interactive prompt loop."""
        self.renderer.print_banner(self.config)

        prompt_tokens = [
            ("class:prompt-bracket", "["),
            ("class:prompt-name", "Sparrow"),
            ("class:prompt-bracket", "]"),
            ("class:prompt-arrow", " ❯ "),
        ]

        while True:
            try:
                # Capture user input using prompt_toolkit
                user_input = self.prompt_session.prompt(
                    prompt_tokens,
                    style=self.prompt_style,
                ).strip()

                if not user_input:
                    continue

                # Check for slash commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue

                # Add user input to conversation session
                self.session.add_message("user", user_input)

                # Format full context prompt
                full_prompt = self.session.format_prompt()

                # Stream response live to terminal using Rich
                live, update_fn = self.renderer.live_stream_markdown()
                response_text = ""

                with live:
                    for chunk in self.generator.stream_generate(
                        prompt=full_prompt,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        on_token_callback=update_fn,
                    ):
                        response_text += chunk

                # Save assistant turn to chat history
                self.session.add_message("assistant", response_text)

            except KeyboardInterrupt:
                # Interrupted by user (Ctrl+C), return to fresh prompt
                self.renderer.console.print("\n[dim]Generation halted by user.[/dim]")
                continue
            except EOFError:
                # User pressed Ctrl+D
                self.renderer.console.print("\n[dim]Exiting Sparrow. Goodbye![/dim]")
                break
            except Exception as e:
                self.renderer.print_error(f"Unexpected error: {str(e)}")
