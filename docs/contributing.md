# Contributing

We welcome contributions to SignalFlow!

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Development Setup
```bash
git clone https://github.com/pathway2nothing/signalflow-trading.git
cd signalflow-trading
pip install -e ".[dev,docs]"
```

## Code Style

- Use `ruff format` for formatting
- Use `ruff` for linting
- Add type hints
- Write Google-style docstrings
- Run `pre-commit run --files <changed files>` before opening a PR (ruff, ruff-format, mypy)

## Testing
```bash
pytest
pytest --cov=signalflow
```

## Documentation
```bash
mkdocs serve
```

## Contact

- **Email**: [pathway2nothing@gmail.com](mailto:pathway2nothing@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/pathway2nothing/signalflow-trading/issues)