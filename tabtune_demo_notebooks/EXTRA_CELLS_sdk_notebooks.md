# Extra cells for the classification and regression SDK notebooks

These cells are already in `01_lending_club_classification_sdk.ipynb` and `02_home_credit_regression_sdk.ipynb`. To add them to your own copy, paste each block as one code cell at the place noted.

Only the second line of block A0 differs between notebooks:

- Classification (Lending Club): `DF_TRAIN, DF_HOLD, TARGET_COL, TASK = train_full, val_full, LABEL, "classification"`
- Regression (Home Credit): `DF_TRAIN, DF_HOLD, TARGET_COL, TASK = X_train, X_val, TARGET, "regression"`

## A0 · Data preparation lab: setup

*Insert after the data dashboard cell and before the sampling cell (regression: after the leakage cell).*

```python
# 🧪 Data preparation lab · setup
DF_TRAIN, DF_HOLD, TARGET_COL, TASK = train_full, val_full, LABEL, "classification"   # regression: DF_TRAIN, DF_HOLD, TARGET_COL, TASK = X_train, X_val, TARGET, "regression"
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PREP_N = 8000                                                   # rows used for the analysis plots
prep = DF_TRAIN.sample(min(PREP_N, len(DF_TRAIN)), random_state=7).reset_index(drop=True)
y_prep = prep[TARGET_COL].astype(float) if TASK == "regression" else prep[TARGET_COL].astype(int)
NUMERIC = [c for c in FEATURES if pd.api.types.is_numeric_dtype(prep[c])]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
print(f"{len(DF_TRAIN):,} training rows · {len(FEATURES)} candidate features "
      f"({len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical) · analysis sample {len(prep):,} rows")
```

## A1 · Data quality profile

*Directly after A0.*

```python
# 🔎 1 · Data quality profile: missing values, cardinality, constant columns, the backend's drop rule
miss = prep[FEATURES].isna().mean().mul(100)
uniq = prep[FEATURES].nunique() / len(prep)
const = [c for c in FEATURES if prep[c].nunique(dropna=True) <= 1]
binary = [c for c in NUMERIC if prep[c].dropna().isin([0, 1]).all()]
would_drop = uniq[uniq > 0.95].index.tolist()                   # platform rule: >95% unique values is treated as an identifier

kinds = pd.Series({"continuous": len(NUMERIC) - len(binary), "binary / one-hot": len(binary), "categorical": len(CATEGORICAL)})
top_miss = miss.sort_values(ascending=False).head(20)[::-1]
fig = make_subplots(rows=1, cols=3, column_widths=[0.22, 0.4, 0.38],
                    subplot_titles=("Feature types", "Most incomplete features (% missing)", "Cardinality: unique values ÷ rows"))
fig.add_trace(go.Bar(x=kinds.index, y=kinds.values, marker_color=[BLUE, TEAL, ORANGE], text=kinds.values, textposition="outside", showlegend=False), 1, 1)
fig.add_trace(go.Bar(x=top_miss.values, y=top_miss.index, orientation="h", marker_color=[RED if v > 50 else GREY for v in top_miss.values], showlegend=False), 1, 2)
fig.add_trace(go.Histogram(x=uniq.values, nbinsx=40, marker_color=BLUE, showlegend=False), 1, 3)
fig.add_vline(x=0.95, line_dash="dot", line_color=RED, row=1, col=3)
fig.update_layout(template="plotly_white", height=520, title="Data quality profile · red line: columns right of it are dropped by the platform as identifiers")
fig.show()
display(pd.DataFrame({"check": ["features with any missing value", "features more than 50% missing", "constant features",
                                "features above the 95%-unique rule", "duplicate rows in the sample"],
                      "count": [int((miss > 0).sum()), int((miss > 50).sum()), len(const), len(would_drop),
                                int(prep[FEATURES].duplicated().sum())]}).style.hide(axis="index"))
```

## A2 · Distributions, skew and outliers

*After A1.*

```python
# 📈 2 · Target and feature distributions: skew, outliers, and what a transform does
cont = [c for c in NUMERIC if c not in binary and prep[c].nunique() > 20]
skew = prep[cont].skew().abs().sort_values(ascending=False)
q1, q3 = prep[cont].quantile(0.25), prep[cont].quantile(0.75)
outlier = (((prep[cont] < q1 - 3 * (q3 - q1)) | (prep[cont] > q3 + 3 * (q3 - q1))).mean() * 100).sort_values(ascending=False)
signed_log = lambda s: np.sign(s) * np.log1p(np.abs(s))
show = skew.head(3).index.tolist()

fig = make_subplots(rows=2, cols=4, vertical_spacing=0.16,
                    subplot_titles=["Target"] + [f"{c[:26]} · raw" for c in show] + ["Most skewed features (|skew|)"] + [f"{c[:26]} · signed log" for c in show])
if TASK == "classification":
    vc = y_prep.value_counts().sort_index()
    fig.add_trace(go.Bar(x=[CLASS_NAMES[i] for i in vc.index], y=vc.values, marker_color=[BLUE, RED], text=vc.values, textposition="outside", showlegend=False), 1, 1)
else:
    fig.add_trace(go.Histogram(x=y_prep, nbinsx=60, marker_color=BLUE, showlegend=False), 1, 1)
for j, c in enumerate(show, 2):
    fig.add_trace(go.Histogram(x=prep[c], nbinsx=60, marker_color=GREY, showlegend=False), 1, j)
    fig.add_trace(go.Histogram(x=signed_log(prep[c]), nbinsx=60, marker_color=TEAL, showlegend=False), 2, j)
s10 = skew.head(10)[::-1]
fig.add_trace(go.Bar(x=s10.values, y=[c[:28] for c in s10.index], orientation="h", marker_color=ORANGE, showlegend=False), 2, 1)
fig.update_layout(template="plotly_white", height=640, title="Distributions · heavy tails before and after a signed-log transform")
fig.show()
print(f"{int((skew > 2).sum())} of {len(cont)} continuous features are heavily skewed (|skew| > 2) · "
      f"{int((outlier > 1).sum())} have more than 1% extreme outliers")
print("Tabular foundation models apply their own robust scaling, so no manual transform is needed before upload; "
      "the plot shows why that scaling matters.")
```

## A3 · Signal and redundancy

*After A2.*

```python
# 🎯 3 · Which features carry signal, and which are redundant
Xn = prep[NUMERIC].replace([np.inf, -np.inf], np.nan)
Xn = Xn.fillna(Xn.median()).fillna(0)
mi_fn = mutual_info_regression if TASK == "regression" else mutual_info_classif
MI = pd.Series(mi_fn(Xn, y_prep, random_state=0), index=NUMERIC).sort_values(ascending=False)
top = MI.head(20).index.tolist()
corr = Xn[top].corr()
full_corr = Xn[MI.head(60).index].corr().abs()
pairs = (full_corr.where(np.triu(np.ones(full_corr.shape, bool), 1)).stack().sort_values(ascending=False))
redundant = pairs[pairs > 0.9]

fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Mutual information with the target (top 20)", "Correlation between those features"))
fig.add_trace(go.Bar(x=MI.head(20).values[::-1], y=[c[:30] for c in top[::-1]], orientation="h", marker_color=BLUE, showlegend=False), 1, 1)
fig.add_trace(go.Heatmap(z=corr.values, x=[c[:18] for c in top], y=[c[:18] for c in top], colorscale="RdBu_r", zmin=-1, zmax=1, showscale=True), 1, 2)
fig.update_layout(template="plotly_white", height=640, title="Signal and redundancy")
fig.show()
print(f"{int((MI < 0.001).sum())} of {len(NUMERIC)} numeric features carry almost no signal · {len(redundant)} highly correlated pairs (|r| > 0.9) among the top 60")
if len(redundant):
    display(redundant.head(8).rename("abs correlation").reset_index().rename(columns={"level_0": "feature A", "level_1": "feature B"}))

# How the target moves across the range of the four strongest features
fig = make_subplots(rows=1, cols=4, subplot_titles=[c[:28] for c in top[:4]])
for j, c in enumerate(top[:4], 1):
    b = pd.qcut(prep[c].rank(method="first"), 10, labels=False)
    g = pd.DataFrame({"b": b, "x": prep[c], "y": y_prep}).groupby("b").agg(x=("x", "median"), y=("y", "mean"))
    fig.add_trace(go.Scatter(x=g.x, y=g.y * (100 if TASK == "classification" else 1), mode="lines+markers", line=dict(color=ORANGE, width=3), showlegend=False), 1, j)
fig.update_yaxes(title_text="default rate (%)" if TASK == "classification" else f"mean {TARGET_COL}", row=1, col=1)
fig.update_layout(template="plotly_white", height=380, title="Target response across deciles of the strongest features")
fig.show()
```

## A4 · Automated feature engineering

*After A3.*

```python
# 🛠️ 4 · Automated feature engineering: generate candidates, keep the ones that add signal, measure the gain
TOP_K, KEEP = 8, 10
base = [c for c in MI.index if c in cont][:TOP_K]
recipes = []                                                      # (name, operation, a, b)
for i, a in enumerate(base):
    for b in base[i + 1:]:
        recipes += [(f"fe_prod__{a}__{b}", "prod", a, b), (f"fe_diff__{a}__{b}", "diff", a, b)]
        if (prep[b] > 0).all():
            recipes.append((f"fe_ratio__{a}__{b}", "ratio", a, b))
recipes.append(("fe_missing_count", "missing", None, None))


def apply_recipes(df, recs):
    out = {}
    for name, op, a, b in recs:
        if op == "prod":
            out[name] = df[a] * df[b]
        elif op == "diff":
            out[name] = df[a] - df[b]
        elif op == "ratio":
            out[name] = df[a] / df[b].where(df[b] > 0)
        elif op == "missing":
            out[name] = df[[c for c in FEATURES if c in df.columns]].isna().sum(axis=1)
    return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)


cand = apply_recipes(prep, recipes)
cand_f = cand.fillna(cand.median()).fillna(0)
cand_mi = pd.Series(mi_fn(cand_f, y_prep, random_state=0), index=cand.columns)
parent_mi = pd.Series({n: max(MI.get(a, 0), MI.get(b, 0)) if a else 0 for n, _, a, b in recipes})
gain = (cand_mi - parent_mi).sort_values(ascending=False)
KEPT = gain.head(KEEP).index.tolist()
KEPT_RECIPES = [r for r in recipes if r[0] in KEPT]

# Probe model: does the engineered set improve a quick cross-validated model?
Model = HistGradientBoostingRegressor if TASK == "regression" else HistGradientBoostingClassifier
scoring = "r2" if TASK == "regression" else "roc_auc"
cv = lambda X: cross_val_score(Model(max_iter=120, random_state=0), X, y_prep, cv=3, scoring=scoring).mean()
score_raw = cv(Xn)
score_fe = cv(pd.concat([Xn, cand_f[KEPT]], axis=1))

fig = make_subplots(rows=1, cols=3, column_widths=[0.36, 0.4, 0.24],
                    subplot_titles=(f"{len(recipes)} candidates: signal vs best parent", "Kept features: information gain over parents", f"Probe model ({scoring}, 3-fold)"))
fig.add_trace(go.Scatter(x=parent_mi, y=cand_mi, mode="markers", showlegend=False,
                         marker=dict(color=[ORANGE if n in KEPT else GREY for n in cand_mi.index], size=[10 if n in KEPT else 5 for n in cand_mi.index])), 1, 1)
mx = float(max(cand_mi.max(), parent_mi.max()))
fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], line=dict(color=DARK, dash="dot"), showlegend=False), 1, 1)
g = gain[KEPT][::-1]
fig.add_trace(go.Bar(x=g.values, y=[n.replace("fe_", "").replace("__", " · ")[:44] for n in g.index], orientation="h", marker_color=ORANGE, showlegend=False), 1, 2)
fig.add_trace(go.Bar(x=["raw features", "+ engineered"], y=[score_raw, score_fe], marker_color=[GREY, ORANGE],
                     text=[f"{score_raw:.4f}", f"{score_fe:.4f}"], textposition="outside", showlegend=False), 1, 3)
fig.update_xaxes(title_text="best parent's mutual information", row=1, col=1); fig.update_yaxes(title_text="candidate's mutual information", row=1, col=1)
fig.update_yaxes(range=[min(score_raw, score_fe) * 0.97, max(score_raw, score_fe) * 1.02], row=1, col=3)
fig.update_layout(template="plotly_white", height=560, title="Automated feature engineering · orange = kept")
fig.show()
print(f"kept {len(KEPT)} engineered features · probe {scoring}: {score_raw:.4f} → {score_fe:.4f} ({score_fe - score_raw:+.4f})")
```

## A5 · Apply the recipe, pipeline summary

*After A4, still before the sampling cell, so the engineered features are uploaded.*

```python
# ✅ 5 · Apply the recipe to the full tables, then summarise the preparation pipeline
ADD_ENGINEERED = score_fe > score_raw          # only ship engineered features if the probe says they help; set True/False to override
print("engineered features", "ADDED: the probe improved" if ADD_ENGINEERED else "NOT added: the probe did not improve, raw features go to the platform")
if ADD_ENGINEERED:
    for _df in (DF_TRAIN, DF_HOLD):
        fe = apply_recipes(_df, KEPT_RECIPES)
        for c in fe.columns:
            _df[c] = fe[c].round(6)
    FEATURES = [c for c in DF_TRAIN.columns if c not in (UID, TARGET_COL, *FEATURE_EXCLUDE)]

n_raw = DF_TRAIN.shape[1] - (len(KEPT) if ADD_ENGINEERED else 0)
stages = [("columns in the source table", n_raw), ("− identifier and target", n_raw - 2),
          ("− excluded (identifiers, leakage)", n_raw - 2 - len(FEATURE_EXCLUDE)),
          ("− constant", n_raw - 2 - len(FEATURE_EXCLUDE) - len(const)),
          ("+ engineered", n_raw - 2 - len(FEATURE_EXCLUDE) - len(const) + (len(KEPT) if ADD_ENGINEERED else 0))]

# 2-D map of the data: do the classes (or target levels) separate?
Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(Xn[MI.head(30).index]))
fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], subplot_titles=("Feature pipeline", "The data in two dimensions (PCA of the 30 strongest features)"))
fig.add_trace(go.Funnel(y=[s for s, _ in stages], x=[v for _, v in stages], marker_color=[GREY, GREY, RED, RED, ORANGE], textinfo="value", showlegend=False), 1, 1)
if TASK == "classification":
    for cls, col in ((0, BLUE), (1, RED)):
        m = (y_prep == cls).to_numpy()
        fig.add_trace(go.Scattergl(x=Z[m, 0], y=Z[m, 1], mode="markers", name=CLASS_NAMES[cls], marker=dict(size=4, color=col, opacity=0.45)), 1, 2)
else:
    fig.add_trace(go.Scattergl(x=Z[:, 0], y=Z[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=4, color=np.log1p(y_prep.clip(lower=0)), colorscale="Viridis", opacity=0.6, colorbar=dict(title="log target"))), 1, 2)
fig.update_layout(template="plotly_white", height=540, title="From source table to model-ready features", legend=dict(orientation="h", y=-0.1))
fig.show()
print(f"{len(FEATURES)} features go to the platform ({len(KEPT) if ADD_ENGINEERED else 0} engineered). "
      "The platform then encodes categoricals, fills missing values and applies each model's own preprocessing.")
```

## B · AutoML race: classic ML vs foundation models

*Insert after the model comparison cell (section 4 or 5).*

```python
# 🤖 AutoML race: classic ML with automatic hyperparameter search vs zero-shot foundation models
RUN_AUTOML = True
CLASSIC = ["XGBoost", "LGBoost", "CatBoost", "RandomForest"]
CLASSIC_COMPUTE = UPLOAD_COMPUTE                                   # classic models train on CPU

def metric_lookup(summary, *names):
    """First matching metric from a model summary, whatever the key spelling."""
    em = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in eval_metrics(summary).items() if isinstance(v, (int, float))}
    for n in names:
        k = re.sub(r"[^a-z0-9]", "", n.lower())
        if k in em:
            return em[k]
    return None

HEADLINE = ("r2_score", "r2") if TASK == "regression" else ("RocAucScore", "roc_auc", "auc")
SECOND = ("RootMeanSquaredError", "rmse") if TASK == "regression" else ("F1Score", "f1")
CLASSIC_BUILDS = {}
if RUN_AUTOML:
    for model in CLASSIC:
        name = train(project, f"{model} · AutoML (Optuna)", model=model, training_strategy="simple", compute_type=CLASSIC_COMPUTE,
                     data_config={"tags": ["Training"], "test_tags": ["Test"], "feature_exclude": FEATURE_EXCLUDE,
                                  "use_optuna": True, "xai_method": ["shap"]})
        if name:
            CLASSIC_BUILDS[model] = name

race = []
for family, builds in (("Foundation model (zero-shot)", BUILDS), ("Classic ML (AutoML)", CLASSIC_BUILDS)):
    for model, name in builds.items():
        s = summary_of(project, name)
        mins = next((r["minutes"] for r in RUNS if r["created"] == name), None)
        race.append({"model": model, "family": family, "build": name, "headline": metric_lookup(s, *HEADLINE),
                     "second": metric_lookup(s, *SECOND), "minutes": mins})
race = pd.DataFrame(race).sort_values("headline", ascending=False).reset_index(drop=True)
display(race.rename(columns={"headline": HEADLINE[0], "second": SECOND[0]}).style.format(precision=3, na_rep="-")
        .set_caption("One leaderboard: every model scored by the platform on the same Test tag"))

FAM = {"Foundation model (zero-shot)": BLUE, "Classic ML (AutoML)": GREY}
fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{HEADLINE[0]} on the Test tag", "Accuracy vs build time (minutes)"))
r = race.sort_values("headline")
fig.add_trace(go.Bar(x=r.headline, y=r.model, orientation="h", marker_color=[FAM[f] for f in r.family], text=r.headline, texttemplate="%{text:.3f}", showlegend=False), 1, 1)
for fam, g in race.groupby("family"):
    fig.add_trace(go.Scatter(x=g.minutes, y=g.headline, mode="markers+text", text=g.model, textposition="top center", name=fam,
                             marker=dict(size=14, color=FAM[fam])), 1, 2)
fig.update_layout(template="plotly_white", height=480, title="AutoML leaderboard · foundation models need no tuning run", legend=dict(orientation="h", y=-0.15))
fig.show()
```

## C · Modelling at scale

*Insert after the advanced builds section (it reuses `ADV` for the distilled student).*

```python
# 🏗️ Modelling at scale: learning curve, build time, scoring throughput, and the distilled student
RUN_SCALE = True
FRACTIONS = [0.25, 0.5]                                           # the full build (1.0) is the one trained above
curve = [{"fraction": 1.0, "build": BUILDS[BEST]}]
if RUN_SCALE:
    for frac in FRACTIONS:
        name = train(project, f"{BEST} · {int(frac * 100)}% of rows", model=BEST, training_strategy="simple", compute_type=COMPUTE,
                     tunning_strategy="inference", model_config=model_config_for(BEST),
                     data_config={**base_data_config(["lime"]), "sample_percentage": frac})
        if name:
            curve.append({"fraction": frac, "build": name})
for c in curve:
    s = summary_of(project, c["build"])
    c["metric"] = metric_lookup(s, *HEADLINE)
    c["rows"] = (modelling_info(s).get("train_shape") or [None])[0]
    c["minutes"] = next((r["minutes"] for r in RUNS if r["created"] == c["build"]), None)
curve = pd.DataFrame(curve).sort_values("fraction")

# Scoring throughput: the foundation model against its distilled student (if one was built)
speed = []
candidates = {BEST: BUILDS[BEST], **{k: v for k, v in ADV.items() if "istil" in k or "PAD" in k}}
for label, name in candidates.items():
    try:
        ensure_inference(project, name)
        t0 = time.time()
        out = project.model_inference(tag="Test", model_name=name, compute_type=COMPUTE)
        speed.append({"model": label, "rows": len(out), "seconds": time.time() - t0})
    except Exception as exc:
        print(f"throughput skipped for {label}: {str(exc)[:100]}")
speed = pd.DataFrame(speed)
if len(speed):
    speed["rows_per_second"] = speed.rows / speed.seconds
ensure_inference(project, BUILDS[BEST])

fig = make_subplots(rows=1, cols=3, subplot_titles=(f"Learning curve · {HEADLINE[0]} vs context rows", "Build time (minutes)", "Batch scoring throughput (rows/second, end to end)"))
x = curve.rows.fillna(curve.fraction).astype(float)
fig.add_trace(go.Scatter(x=x, y=curve.metric, mode="lines+markers", line=dict(color=BLUE, width=3), showlegend=False), 1, 1)
fig.add_trace(go.Bar(x=[f"{int(f * 100)}%" for f in curve.fraction], y=curve.minutes, marker_color=GREY, showlegend=False), 1, 2)
if len(speed):
    fig.add_trace(go.Bar(x=speed.model, y=speed.rows_per_second, marker_color=[BLUE] + [ORANGE] * (len(speed) - 1),
                         text=speed.rows_per_second, texttemplate="%{text:.0f}", showlegend=False), 1, 3)
fig.update_layout(template="plotly_white", height=440, title=f"Scale profile · {BEST}")
fig.show()
display(pd.DataFrame({"lever": ["Context size", "Row sampling", "Compute", "Ensembles", "Distillation", "Batch scoring"],
                      "how it is used here": ["foundation models read the training rows as context; the platform caps a build at 10,000 rows",
                                              "sample_percentage in data_config sets the share of rows per build",
                                              f"GPU queue {COMPUTE} for foundation models, CPU {UPLOAD_COMPUTE} for classic models and uploads",
                                              "several foundation models combined by greedy selection",
                                              "a tree or MLP student learns the foundation model's predictions for cheap, fast scoring",
                                              "model_inference scores a whole tag in one call"]}).style.hide(axis="index"))
```

## D1 · Retraining agent

*Insert after block C, before the run-log cell.*

```python
# 🔁 Retraining agent: observe → decide → act → verify → promote
# A small policy agent. It watches a new batch for drift and decay, decides whether to retrain,
# trains a challenger on the platform, checks it on fresh data and promotes it only if it wins.
RUN_AGENT = True
PSI_ALERT, DECAY_ALERT, PROMOTE_MARGIN = 0.20, 0.02, 0.003


def psi(ref, new, bins=10):
    """Population stability index of one feature: 0 = identical, above 0.2 = material drift."""
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref.dropna(), edges)[0] / max(ref.notna().sum(), 1) + 1e-4
    b = np.histogram(new.dropna(), edges)[0] / max(new.notna().sum(), 1) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


def score_frame(scored_df, truth):
    """Headline metric from a platform inference frame (probability or prediction column) against the truth."""
    m = scored_df.merge(truth[[UID, TARGET_COL]].rename(columns={TARGET_COL: "_truth"}), on=UID)
    if TASK == "regression":
        return float(skm.r2_score(m["_truth"].astype(float), m["Predicted_value_AutoML"].astype(float)))
    return float(skm.roc_auc_score((m["_truth"].astype(str) == "1").astype(int), m["pred_proba_AutoML"].astype(float)))


class RetrainAgent:
    def __init__(self, project, champion_model, champion_build):
        self.project, self.model, self.champion, self.challenger = project, champion_model, champion_build, None
        self.trace, self.state = [], {}

    def log(self, step, finding, decision=""):
        self.trace.append({"step": step, "finding": finding, "decision": decision, "at": pd.Timestamp.now(), "seconds": 0.0})
        print(f"[{step}] {finding}" + (f" → {decision}" if decision else ""))

    def _timed(self, fn, *a, **k):
        t0 = time.time(); out = fn(*a, **k); self.trace[-1]["seconds"] = round(time.time() - t0, 1); return out

    def _upload(self, df, tag):
        path = f"/content/agent_{tag.lower()}.csv"
        df.to_csv(path, index=False)
        upload_once(self.project, path, tag)

    def _score(self, build, tag, truth):
        ensure_inference(self.project, build)
        return score_frame(self.project.model_inference(tag=tag, model_name=build, compute_type=COMPUTE), truth)

    def observe(self, reference, batch, batch_tag, baseline):
        num = [c for c in FEATURES if c in batch.columns and pd.api.types.is_numeric_dtype(batch[c]) and reference[c].nunique() > 10]
        self.state["psi"] = pd.Series({c: psi(reference[c], batch[c]) for c in num}).sort_values(ascending=False)
        drifted = int((self.state["psi"] > PSI_ALERT).sum())
        self.log("observe · drift", f"{drifted} of {len(num)} features above PSI {PSI_ALERT} (max {self.state['psi'].iloc[0]:.2f} on {self.state['psi'].index[0]})")
        self.log("observe · performance", "scoring the new batch with the champion")
        self._timed(self._upload, batch, batch_tag)
        live = self._score(self.champion, batch_tag, batch)
        self.state.update(baseline=baseline, live=live, drifted=drifted, batch_tag=batch_tag)
        self.trace[-1]["finding"] = f"champion {HEADLINE[0]} {baseline:.3f} on Test → {live:.3f} on the new batch ({live - baseline:+.3f})"
        print("   ", self.trace[-1]["finding"])
        return self

    def decide(self):
        s = self.state
        reasons = ([f"{s['drifted']} drifted features"] if s["drifted"] >= 1 else []) + \
                  ([f"performance down {s['baseline'] - s['live']:.3f}"] if s["baseline"] - s["live"] > DECAY_ALERT else [])
        s["retrain"] = bool(reasons)
        self.log("decide", "; ".join(reasons) or "no material drift or decay", "RETRAIN" if reasons else "KEEP CHAMPION")
        return self

    def act(self):
        if not self.state["retrain"]:
            return self
        self.log("act", f"training a challenger {self.model} on Training + {self.state['batch_tag']}")
        cfg = {**base_data_config(xai_for(self.model, TASK)), "tags": ["Training", self.state["batch_tag"]]}
        self.challenger = self._timed(train, self.project, f"{self.model} · challenger", model=self.model, training_strategy="simple",
                                      compute_type=COMPUTE, tunning_strategy="inference", data_config=cfg, model_config=model_config_for(self.model))
        self.trace[-1]["decision"] = self.challenger or "build failed"
        return self

    def verify(self, recent, recent_tag):
        if not self.challenger:
            return self
        self.log("verify", f"scoring champion and challenger on unseen recent data ({len(recent):,} rows)")
        self._upload(recent, recent_tag)
        t0 = time.time()
        self.state["champ_recent"] = self._score(self.champion, recent_tag, recent)
        self.state["chall_recent"] = self._score(self.challenger, recent_tag, recent)
        self.trace[-1]["seconds"] = round(time.time() - t0, 1)
        self.trace[-1]["finding"] = f"champion {self.state['champ_recent']:.3f} vs challenger {self.state['chall_recent']:.3f} on recent data"
        print("   ", self.trace[-1]["finding"])
        return self

    def promote(self):
        s = self.state
        win = bool(self.challenger) and s.get("chall_recent", -np.inf) - s.get("champ_recent", np.inf) >= PROMOTE_MARGIN
        if win:
            self.project.activate_model(self.challenger)
        else:
            self.project.activate_model(self.champion)
        s["promoted"] = win
        self.log("promote", f"margin required {PROMOTE_MARGIN}", f"PROMOTED {self.challenger}" if win else f"KEPT {self.champion}")
        return self


# --- New data arrives: unseen rows whose mix has shifted towards one end of the strongest feature -------------
used = set(test_sample[UID])
pool = DF_HOLD[~DF_HOLD[UID].isin(used)]
drift_feature = next(c for c in MI.index if c in pool.columns and pool[c].nunique() > 20)
z = (pool[drift_feature] - pool[drift_feature].mean()) / (pool[drift_feature].std() + 1e-9)
shifted = pool.sample(min(2400, len(pool)), weights=np.exp(1.2 * z.fillna(0).clip(-3, 3)), random_state=5)   # real rows, shifted mix
new_batch, recent = shifted.iloc[:1200].copy(), shifted.iloc[1200:].copy()
print(f"new batch: {len(new_batch):,} rows · recent hold-out: {len(recent):,} rows · mix shifted on '{drift_feature}'")

if RUN_AGENT:
    baseline = score_frame(scored, test_sample)
    agent = RetrainAgent(project, BEST, BUILDS[BEST])
    agent.observe(train_sample, new_batch, "Batch1", baseline).decide().act().verify(recent, "Recent1").promote()
```

## D2 · Retraining agent dashboard

*Directly after D1.*

```python
# 📊 Retraining agent · what it saw, what it decided, what changed
if RUN_AGENT:
    s, tr = agent.state, pd.DataFrame(agent.trace)
    top_psi = s["psi"].head(12)[::-1]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.18, specs=[[{}, {}], [{}, {}]],
                        subplot_titles=("Drift by feature (PSI)", f"Most drifted feature · {s['psi'].index[0][:34]}",
                                        f"{HEADLINE[0]}: champion vs challenger", "Agent timeline (seconds per step)"))
    fig.add_trace(go.Bar(x=top_psi.values, y=[c[:30] for c in top_psi.index], orientation="h", showlegend=False,
                         marker_color=[RED if v > PSI_ALERT else ORANGE if v > 0.1 else GREY for v in top_psi.values]), 1, 1)
    fig.add_vline(x=PSI_ALERT, line_dash="dot", line_color=RED, row=1, col=1)
    f0 = s["psi"].index[0]
    fig.add_trace(go.Histogram(x=train_sample[f0], histnorm="probability", nbinsx=40, name="training data", marker_color=GREY, opacity=0.7), 1, 2)
    fig.add_trace(go.Histogram(x=new_batch[f0], histnorm="probability", nbinsx=40, name="new batch", marker_color=RED, opacity=0.6), 1, 2)
    bars = [("champion · Test", s["baseline"], GREY), ("champion · new batch", s["live"], RED)]
    if "chall_recent" in s:
        bars += [("champion · recent", s["champ_recent"], GREY), ("challenger · recent", s["chall_recent"], BLUE)]
    fig.add_trace(go.Bar(x=[b[0] for b in bars], y=[b[1] for b in bars], marker_color=[b[2] for b in bars],
                         text=[f"{b[1]:.3f}" for b in bars], textposition="outside", showlegend=False), 2, 1)
    lo = min(b[1] for b in bars)
    fig.update_yaxes(range=[lo - abs(lo) * 0.05 - 0.02, max(b[1] for b in bars) + 0.03], row=2, col=1)
    fig.add_trace(go.Bar(x=tr.seconds, y=tr.step, orientation="h", marker_color=TEAL, text=tr.decision, textposition="auto", showlegend=False), 2, 2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    verdict = f"PROMOTED {agent.challenger}" if s.get("promoted") else f"KEPT {agent.champion}"
    fig.update_layout(template="plotly_white", height=820, barmode="overlay", legend=dict(orientation="h", y=1.08),
                      title=f"Retraining agent · outcome: {verdict}")
    fig.show()
    display(tr[["step", "finding", "decision", "seconds"]].style.hide(axis="index").set_caption("Agent decision log (auditable)"))
```
