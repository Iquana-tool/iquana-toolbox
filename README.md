# Common Pydantic Schemas
This repo contains common pydantic schemas used by every service and the main backend. You can install this repo and import it into custom services making it much easier to transfer data between services and the backend, because you do not have to redefine your pydantic models.

## Scope: no deep-learning runtime

This package is installed by *every* consumer — including the backend,
which does not run models directly. It therefore must stay free
of `torch`, `torchvision` and `transformers`: a transitive torch drags the entire
CUDA toolkit (nvidia-*, triton, ~GBs) into every `uv sync`.

What lives here: pydantic schemas, the quantification metric registry, the MLflow
model registry, and `ai.base_classes` — the *abstract* model interface (no compute).

What does not: model implementations, backbones and dataloaders. Those belong in
`ai-service` under `models/` (`models.backbones.dinov3`, `models.dataloaders`).
