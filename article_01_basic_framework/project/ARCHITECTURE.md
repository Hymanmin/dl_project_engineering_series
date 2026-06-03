# Architecture

This template follows a registry-driven modular design.

## Directories

- `main.py`: unified command entry for train and predict modes.
- `config/`: YAML configs and config loader with `_base_` inheritance and CLI overrides.
- `datatxt/`: dataset list files.
- `launcher/`: train and predict workflow definitions.
- `core/`: reusable components, including registries, datasets, networks, losses, and models.
- `core/networks/`: network framework split into backbone, neck, and head.
- `utils/`: shared utilities.

## Extension Flow

1. Add a new dataset under `core/dataset/<task>/` and register it with `DATASETS`.
2. Add a new network under `core/networks/<task>/` and register the framework with `NETWORKS`.
3. Add a new model under `core/models/<task>/` and register it with `MODELS`.
4. Add or reuse a loss under `core/losses/` and register it with `LOSSES`.
5. Create train and predict configs under `config/`.


