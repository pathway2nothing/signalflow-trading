"""
SignalFlow CLI - Command-line interface for backtest execution.

Usage:
    sf run config.yaml        Run backtest from YAML config
    sf list detectors         List available detectors
    sf list metrics           List available metrics
    sf validate config.yaml   Validate config without running
    sf init                   Create sample config file
"""

from signalflow.cli.main import cli

__all__ = ["cli"]
