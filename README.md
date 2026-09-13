# iquana-toolbox

The shared Python package of [IQUANA](https://github.com/Iquana-tool/iquana-tool) —
**I**ntelligent **QU**antification, **AN**notation and **A**nalysis, a tool for AI-assisted
segmentation, annotation and quantification of scientific image datasets, built at
[DFKI](https://www.dfki.de/).

It holds everything more than one IQUANA component needs to agree on: the Pydantic schemas
that travel between them, the quantification metric registry, the MLflow model registry, and
the abstract model interface. Installing it into a custom service means the wire formats do
not have to be redefined on both sides.

- **User documentation:** https://iquana-tool.github.io/docs/
- **Issues:** all IQUANA bug reports and feature requests go to
  [iquana-tool/issues](https://github.com/Iquana-tool/iquana-tool/issues/new/choose).

---

## Scope: no deep-learning runtime

This package is installed by *every* consumer — including the backend, which does not run
models directly. It must therefore stay free of `torch`, `torchvision` and `transformers`: a
transitive torch drags the entire CUDA toolkit (nvidia-*, triton, several GB) into every
`uv sync`.

**What lives here:** Pydantic schemas, the quantification metric registry, the MLflow model
registry, inference overlap primitives, caches, and `ai.base_classes` — the *abstract* model
interface, with no compute.

**What does not:** model implementations, backbones and dataloaders. Those belong in
[ai-service](https://github.com/Iquana-tool/ai-service) under `models/`
(`models.backbones.dinov3`, `models.dataloaders`), together with the capability mixins that
extend `ai.base_classes`.

The permission matrix is likewise deliberately *not* here — it lives in the backend, because
this package is consumed as a git-pinned dependency and a permission change would otherwise
need a toolbox release and a re-pin in every consumer.

---

## What is in it

```
src/iquana_toolbox/
├── schemas/
│   ├── database/         # Contours, masks, images, datasets, labels, contour hierarchy,
│   │                     #   quantification and quantification profiles
│   ├── networking/
│   │   ├── http/         # Service request/response envelopes
│   │   └── websockets/   # The annotation-session protocol messages
│   ├── input_contract.py # Model-declared inference input contracts
│   ├── model_info.py     # ModelInfo and its per-task subclasses
│   ├── prompts.py        # Point, box, polygon and freedraw prompts
│   ├── scale.py, training.py, user.py
├── quantification/
│   ├── registry.py       # Metric base class, Tier, UnitKind, METRIC_REGISTRY
│   ├── context.py        # Per-image QuantContext handed to every metric
│   ├── geometry_math.py
│   └── metrics/          # geometry, appearance, relational, contextual
├── inference/overlap.py  # NMS deduplication and containment-based parent finding
├── ai/base_classes.py    # Abstract model interface (BaseModel over mlflow.pyfunc)
├── mlflow.py             # MLFlowModelRegistry — register, list and load models
└── caches.py             # Image and TTL caches shared by the services
```

### Quantification metrics

Every metric is a small class registered under a unique string key. The registry is the
single source of truth for which metrics exist, what they measure (**tier**), how their
values relate to physical units (**unit kind**) and how many components a value has (e.g. 3
for a LAB colour). Metrics compute in batch over a per-image `QuantContext`.

| Tier | Needs | Metrics |
|---|---|---|
| `geometry` | Contour points only | `area`, `perimeter`, `circularity`, `max_diameter` |
| `appearance` | The image pixels | `mean_color_rgb`, `mean_color_lab`, `mean_intensity` |
| `relational` | Sibling / parent contours | `n_children` |
| `contextual` | The whole image context | `nn_distance`, `mean_knn_distance` |

`UnitKind` decides how a value is labelled once an image has a pixel scale: `LENGTH` values
take the image unit (`mm`), `AREA` values its square (`mm²`), everything else is unitless.

Adding a metric is a subclass plus `@register_metric`; importing
`iquana_toolbox.quantification.metrics` fires every registration.

### Input contracts

An `InputContract` is the model-owned description of the inputs for one advertised task. It
reuses the `HyperParameter` descriptor so the frontend can render training and inference
controls from the same shape. The schema is strict about semantic combinations on purpose:
field validation can say that `max_units` is an integer, but only the cross-field validators
can say that a `none` contract must not claim to consume instances, or that an embedding
contract must name the embedding kinds it needs.

### Overlap primitives

Two batch-inference problems reduce to *how much do these two contours overlap*:

- **Duplicates** — several models may be pointed at the same label, and a patching run adds
  predictions on top of existing annotations. Greedy NMS keeps the highest-scoring proposal
  of each overlapping cluster.
- **Hierarchy** — a child-level prediction arrives with no `parent_id`. Its parent is the
  written contour that *contains* it best. Containment rather than IoU, because a small
  child inside a large parent always has a low IoU.

---

## Installing

Consumers pin a git revision rather than a released version:

```toml
[tool.uv.sources]
iquana-toolbox = { git = "https://github.com/Iquana-tool/iquana-toolbox.git", rev = "<sha>" }
```

For local work on the toolbox alongside a consumer, swap in a path source:

```toml
iquana-toolbox = { path = "../iquana-toolbox", editable = true }
```

**Bumping the pin is a two-repo change.** A consumer's venv keeps serving the synced copy of
the pinned revision, so local toolbox edits are invisible until the pin moves and `uv sync`
runs. When the two are out of step in either direction, the mismatch shows up as an
`AttributeError` or `ModuleNotFoundError` on whatever is newer — check which side is ahead
before assuming the code is broken.

---

## Tests

```bash
uv run pytest tests/ -q
```

`tests/test_mlflow_registry.py` needs a live MLflow server; without one those tests spend
several minutes on connection timeouts before failing. Skip them with
`--ignore=tests/test_mlflow_registry.py`.

---

## Related repositories

| Repo | Role |
|---|---|
| [iquana-tool](https://github.com/Iquana-tool/iquana-tool) | Installer, launcher and the issue tracker for all of IQUANA |
| [backend](https://github.com/Iquana-tool/backend) | REST + WebSocket API, database, exports |
| [frontend-react](https://github.com/Iquana-tool/frontend-react) | The web UI |
| [ai-service](https://github.com/Iquana-tool/ai-service) | Model inference and training |

---

## License

AGPL-3.0 — see [LICENSE](LICENSE).
