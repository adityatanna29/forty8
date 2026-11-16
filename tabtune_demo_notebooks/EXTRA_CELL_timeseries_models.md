# Model catalogue cell for the time-series notebooks

Paste as one code cell directly after the imports cell (it needs `pd` and TabTune installed).

```python
# 📚 Time-series models available in TabTune today (read live from the library's registry)
from tabtune.registry import list_time_series_models

def model_kind(spec):
    if spec.name.startswith("TabularTS") or spec.name == "TabPFN-TS":
        return "Tabular model as forecaster"
    return "Foundation model" if spec.default_checkpoint else "Statistical baseline"

TS_MODELS = {
    spec.name: {
        "kind": model_kind(spec),
        "family": spec.family,
        "tasks": sorted(spec.tasks),
        "strategies": sorted(spec.strategies),          # inference = zero-shot, finetune, peft = LoRA
        "default_checkpoint": spec.default_checkpoint or "none (no weights)",
        "max_context": spec.max_context,
        "multivariate": spec.supports_multivariate,
        "covariates": spec.supports_covariates,
        "licence": spec.license.name,
        "commercial_use": {True: "yes", False: "no", None: "unverified"}[spec.license.commercial_use_ok],
    }
    for spec in list_time_series_models()
}

catalogue_ts = pd.DataFrame(TS_MODELS).T.rename_axis("model")
print(f"{len(TS_MODELS)} models registered:", catalogue_ts["kind"].value_counts().to_dict())
for kind in ("Foundation model", "Tabular model as forecaster", "Statistical baseline"):
    part = catalogue_ts[catalogue_ts.kind == kind].drop(columns="kind")
    part = part.assign(tasks=part.tasks.map(", ".join), strategies=part.strategies.map(", ".join))
    display(part.style.set_caption(f"{kind} · {len(part)} models"))
TS_MODELS["Chronos2"]          # one entry of the dictionary
```
