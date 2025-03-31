from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskProgressColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.text import Text


CONSOLE = Console(width=120)

class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text(f"? {self.suffix}", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")

def get_status(msg: str, spinner: str = "arc"):
    return CONSOLE.status(msg, spinner=spinner)

def get_progress(description: str, suffix: Optional[str] = None, show_percent: bool = False):
    """Helper function to return a rich Progress object."""
    progress_list = [
        TextColumn(description), 
        BarColumn(), 
        TaskProgressColumn(show_speed=True) if show_percent else MofNCompleteColumn(),
        ]
    progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
    progress_list += [TimeRemainingColumn(elapsed_when_finished=True, compact=True)]
    progress = Progress(*progress_list)
    return progress