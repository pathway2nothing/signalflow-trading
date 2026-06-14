# CLI

Command-line interface for running flows and managing model artifacts.

!!! info "Entry Point"
    `sf` -- installed automatically with the package via `pip install signalflow-trading`.

## Commands

### `sf run`

Run a flow from a YAML configuration file.

```bash
sf run flow.yaml
```

### `sf list`

List components registered in the SignalFlow registry.

```bash
sf list
```

### `sf promote`

Promote a model artifact between stages.

```bash
sf promote --help
```

### `sf version`

Print the installed package version.

```bash
sf version
```

## API Reference

::: signalflow.cli.main.main
    options:
      show_root_heading: true
      show_source: false
