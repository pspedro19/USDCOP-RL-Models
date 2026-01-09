"""
USD/COP RL Trading System V11 - Logger
=======================================

Session logging with file output.
"""

import time
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Session logger with file output.

    Logs messages to both console and file with timestamps.

    Parameters
    ----------
    logs_dir : Path or str
        Directory for log files
    prefix : str
        Prefix for log file name
    """

    def __init__(self, logs_dir: Path, prefix: str = "v11"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file = self.logs_dir / f"{prefix}_session_{self.session}.log"
        self.cell_start = None

    def log(self, msg: str, level: str = 'INFO'):
        """
        Log a message.

        Parameters
        ----------
        msg : str
            Message to log
        level : str
            Log level (INFO, WARN, ERROR)
        """
        ts = datetime.now().strftime('%H:%M:%S')
        entry = f"[{ts}] [{level:5}] {msg}"

        print(entry)

        with open(self.file, 'a', encoding='utf-8') as f:
            f.write(entry + '\n')

    def header(self, title: str):
        """Log a header."""
        self.log("=" * 70)
        self.log(title.center(70))
        self.log("=" * 70)

    def subheader(self, title: str):
        """Log a subheader."""
        self.log("")
        self.log(f"--- {title} ---")

    def start_cell(self, num: int, name: str):
        """
        Mark the start of a cell/section.

        Parameters
        ----------
        num : int
            Cell/section number
        name : str
            Cell/section name
        """
        self.cell_start = time.time()
        self.log("")
        self.header(f"CELDA {num}: {name}")

    def end_cell(self, num: int, summary_items: list):
        """
        Mark the end of a cell/section.

        Parameters
        ----------
        num : int
            Cell/section number
        summary_items : list
            List of summary items to log
        """
        elapsed = time.time() - self.cell_start if self.cell_start else 0
        self.log("")
        self.log("-" * 50)
        self.log(f"RESUMEN CELDA {num} ({elapsed:.1f}s):")
        for item in summary_items:
            self.log(f"  >> {item}")
        self.log("=" * 70)

    def separator(self, char: str = "=", length: int = 60):
        """Log a separator line."""
        self.log(char * length)

    def blank(self):
        """Log a blank line."""
        self.log("")
