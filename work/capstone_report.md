# Capstone Report — Ranking Signal Analysis

* **Author:** Shreshth114
* **Lane:** Ranking Signal Analysis
* **Repo:** https://github.com/Shreshth114/flyrank-ml-internship
* **Date:** 2026-08-30

---

## 0. Abstract

Which content and search performance signals are most predictive of a page achieving and sustaining a high search ranking position — and which commonly assumed signals have no real predictive power? We used the FlyRank internship-starter dataset: 30,000 anonymized content-performance rows across 32 pseudonymized clients, aggregated over a 90-day window. We trained a Random Forest classifier (GroupShuffleSplit on client_id, test_size=0.2) to predict a binary top-performer label, then applied SHAP TreeExplainer to rank signal importance. The model achieves Precision@50 = 0.90 versus a majority-class baseline of 0.18, a 5× lift; impression_momentum is the dominant signal (SHAP 0.1418, 4.5× the second-ranked feature), while search_volume shows a correlation of 0.0012 with actual impressions and word_count shows no meaningful difference between top and bottom performers. The output is a ranked signal report and six-action playbook for content strategists: which signals to track, which pages to prioritize, and which commonly used metrics to stop relying on.

---

## 1. Problem Framing

**Decision this supports:** A content strategist with a portfolio of pages needs to know which pages and signals to prioritize to improve search ranking tier and arrest declining trends.

**Unit of analysis:** One content page (one row = one content item, 90-day aggregated performance).

**Output:** A ranked signal importance report (SHAP) and action playbook — six recommendations ordered by predictive power.

**Action a human takes:** Given the signal rankings, a content team can:
- Identify pages with high impression_momentum and invest in them before they plateau
- Flag zero-click pages (>500 impressions, 0 clicks) for immediate title/meta fix
- Deprioritize search_volume as a content selection criterion and replace with actual impressions data

**Cost of a wrong call:**
- False positive (flag a non-performer as top performer): wasted content investment
- False negative (miss a top performer): missed compounding growth opportunity

Precision@50 is the primary metric because a content team acts on the top ~50 flagged pages; high precision there means low wasted effort.

**Why ML helps:** Rule-based heuristics (e.g. "pages with search_volume > X are worth targeting") fail because they rely on assumed proxies that turn out to have near-zero correlation with actual outcomes in this dataset. A tree-based model across 38 features can capture non-linear interactions (e.g. impression_momentum matters more when baseline impressions are already above a threshold) that no manual rule system captures.

---

## 2. Data Safety

**Dataset used:** `FlyRank/internship-starter` — public, no authentication required.
30,000 rows, 53 columns, 32 pseudonymized clients. No URLs, titles, keywords, domains, or client names anywhere in the data.

**Columns deliberately excluded as features:**

| Column | Reason |
|---|---|
| `health_score` | Derived from outcome — direct label leakage |
| `needs_ctr_fix` | Derived from outcome — direct label leakage |
| `is_quick_win` | Derived from outcome — direct label leakage |
| `needs_indexing` | Derived from outcome — direct label leakage |
| `needs_engagement_fix` | Derived from outcome — direct label leakage |
| `ai_opportunity` | Derived from outcome — direct label leakage |
| `is_underperformer` | Derived from outcome — direct label leakage |
| `is_declining` | Used to build target — direct label leakage |
| `is_initial_refresh_candidate` | Derived from outcome — direct label leakage |
| `ctr` | Directly encodes `position_tier` used to build target |
| `avg_position` | Directly encodes `position_tier` used to build target |
| `impression_tier` | Ordinal encoding of impressions used in target definition |
| `scroll_rate` | Values >100 — data quality issue |
| `ai_traffic_pct` | Values >100 — data quality issue |
| `age_tier_order` | Ordinal duplicate of `age_tier` |
| `trend_direction` | Used to build target |
| `position_tier` | Used to build target |
| `trend_pct` | Directly encodes trend magnitude used in target |
| `content_id` | Pseudonymous ID — grouping/joining only, never a feature |
| `client_id` | Pseudonymous ID — used for GroupShuffleSplit only, never a feature |
| `provider_used` | High-cardinality string, not numeric |
| `model_used` | High-cardinality string, not numeric |

**Leakage checks performed:**
- First run with ctr/avg_position included → Precision@50 = 1.00, accuracy = 0.98. Confirmed leakage.
- After removing position/trend encodings → Precision@50 = 0.90, accuracy = 0.83. Clean.
- GroupShuffleSplit ensures no client leaks between train and test.

**Client-identifying data:** Nothing in `work/` contains client names, domains, URLs, or any identifying information. All IDs are hashed. No row-level outputs are published — only aggregate statistics and model artifacts.

---

## 3. Baseline

**Baseline:** `DummyClassifier(strategy='most_frequent')` — always predicts the majority class (is_top_performer = 0).

**Why it's a fair comparison:** Evaluated on the same GroupShuffleSplit test set as the model. The base rate of top performers is 17.1% (5,143 / 30,000), so the majority-class classifier achieves 79% accuracy by always predicting 0 — making accuracy a misleading metric. Precision@50 on this baseline = 0.18, which reflects the underlying positive rate in the test set.

**Baseline numbers on test set:**
- Precision@50: 0.18
- Precision (class 1): 0.00
- Recall (class 1): 0.00
- F1 (class 1): 0.00
- Accuracy: 0.79
- Base rate (positive class): 20.7% in test set

---

## 4. Model / Analysis

**Method:** RandomForestClassifier

**Why it fits the lane:** Ranking Signal Analysis requires interpretable feature importance, not just predictions. Random Forest + SHAP TreeExplainer gives per-feature importance values with directional information. It also handles mixed numeric/categorical features and is robust to the scale differences across features (impressions in thousands, rates as decimals).

**Exact hyperparameters:**
```
n_estimators=50, max_depth=6, min_samples_leaf=20,
class_weight='balanced', random_state=42, n_jobs=-1
```

**Target definition (one sentence):**
`is_top_performer = 1` when `position_tier ∈ {top_3, page_1}` AND `trend_direction ∈ {up, stable, flat}`.

**Final feature list (38 features):**
```
search_volume, competition, competition_level, cpc, content_type, main_intent,
word_count, char_count, impressions_90d, clicks_90d, pageviews_90d,
sessions_90d, users_90d, engaged_sessions_90d, ai_sessions_90d,
scroll_events_90d, days_with_impressions, days_with_sessions,
impressions_last_30d, clicks_last_30d, sessions_last_30d,
impressions_prev_30d, clicks_prev_30d, sessions_prev_30d,
content_age_days, age_tier, days_since_last_update, freshness_tier,
word_count_tier, char_count_tier, engagement_rate,
impression_momentum, click_momentum, click_through_efficiency,
engagement_density, session_per_click, content_density, ai_traffic_share
```

**Deliberately left out:** All columns listed in Section 2 above.

---

## 5. Evaluation

**Split:** GroupShuffleSplit on `client_id`, test_size=0.2, random_state=42.
- Train: 23,837 rows | Test: 6,163 rows
- Train positive rate: 16.22% | Test positive rate: 20.70%

**Why grouped by client:** A random split would leak client-specific patterns (writing style, domain authority, niche) into the test set. Grouping by client ensures the model is evaluated on clients it has never seen — a harder and more honest test.

**Primary metric:** Precision@50 — the fraction of actual top performers in the top 50 model-predicted positives. This is the operational metric: a content team acts on the pages the model is most confident about.

**Results:**

| Metric | Baseline | Random Forest |
|---|---|---|
| Precision@50 | 0.18 | **0.90** |
| Precision (class 1) | 0.00 | 0.55 |
| Recall (class 1) | 0.00 | 0.86 |
| F1 (class 1) | 0.00 | 0.67 |
| Accuracy | 0.79 | 0.83 |
| Base rate (test) | 20.7% | 20.7% |

**Error analysis:**
- High recall (0.86) means the model catches most actual top performers — low false negative rate. Good for a discovery task.
- Moderate precision (0.55) on the full positive class means roughly 1 in 2 predicted positives is wrong at threshold 0.5. But Precision@50 = 0.90 — at high confidence scores the model is very reliable.
- False positives tend to be pages with high impression_momentum but poor click conversion — pages growing in visibility but not yet translating to clicks. These are borderline cases, not junk predictions.

---

## 6. Interpretation

**What the model found:**

1. **impression_momentum dominates** (SHAP 0.1418) — 4.5× the second-ranked feature. The single strongest predictor of top performance is whether a page's impressions grew in the last 30 days vs the prior 30 days. This is a momentum signal, not a content quality signal.

2. **Impression and click volume signals fill ranks 2–9** — impressions_prev_30d, impressions_last_30d, click_momentum, clicks_last_30d, clicks_90d all rank highly. Top performers have 2.36× higher median impressions_90d than bottom performers.

3. **click_through_efficiency (rank 7)** — captures pages that have impressions but zero clicks. 2,522 pages (8.4%) have >500 impressions and 0 clicks.

4. **days_since_last_update (rank 14)** — operationally important despite modest SHAP. Stale content correlates with declining trend.

**Surprises and negative results:**

- **search_volume: r = 0.0012 with impressions_90d.** Effectively zero. This is the most actionable negative result — a metric widely used for content prioritization has no correlation with actual traffic in this dataset.

- **word_count: top 2,797 vs bottom 2,902 median.** No meaningful difference, wrong direction. Content length does not predict ranking.

- **engagement_rate and engagement_density both median = 0** across all rows. These signals are flat and carry no predictive information. The 75th percentile of engagement_rate is only 1.35 — nearly the entire dataset sits at zero.

- **competition and cpc (from keyword tools): 8.2% null rate.** Clients without keyword tool access have null values here. These features rank low in SHAP importance (competition rank 25+), confirming keyword tool data is not the primary signal source.

---

## 7. Recommendation

**Ranked actions for a FlyRank content editor:**

| Rank | Signal | Action | Confidence |
|---|---|---|---|
| 1 | impression_momentum | Find pages with momentum > 1.2 and below-median impressions (3,228 pages). Refresh + internal links. These are compounding. | High — #1 SHAP signal by 4.5× |
| 2 | Impression volume trend | Pages losing ground (impressions_prev_30d > impressions_last_30d) need immediate diagnosis. | High — ranks 2–3 in SHAP |
| 3 | click_momentum | Pages with growing impressions but flat clicks need CTR optimization (title/meta). | High — ranks 5–6 in SHAP |
| 4 | click_through_efficiency | 2,522 pages with >500 impressions and 0 clicks are immediate title/meta fix candidates. | High — directly measurable, no model needed |
| 5 | days_since_last_update | Build a freshness queue: pages not updated in 60+ days with flat/declining momentum. | Medium — rank 14 in SHAP, directional |
| 6 | Stop using search_volume + word_count | Remove from content briefs. Replace with impressions + momentum. | High — both confirmed as near-zero signal |

**How to use tomorrow:** Run the notebook on any new data snapshot. Filter `signal_rankings.csv` for features with mean_shap > 0.01. Cross-reference with `impression_momentum > 1.2` to find growing pages, and `clicks_90d == 0 & impressions_90d > 500` for zero-click opportunities.

**Limits:** This is directional, not prescriptive. The model identifies correlates, not causes. A page with high impression_momentum may be growing for reasons entirely outside content control (algorithm update, competitor dropped out, seasonal trend). Act on the signal but investigate the reason.

---

## 8. Reproducibility

**Environment:**
```
python >= 3.10
datasets
pandas
numpy
scikit-learn
shap
matplotlib
seaborn
```

**To re-run from a fresh clone:**
```bash
git clone https://github.com/Shreshth114/flyrank-ml-internship
cd flyrank-ml-internship
pip install datasets pandas scikit-learn shap matplotlib seaborn
jupyter notebook work/notebooks/capstone.ipynb
# Run all cells top to bottom
```

**No authentication needed** — FlyRank/internship-starter is a public dataset.

**Random seeds:** `SEED = 42` used for GroupShuffleSplit, RandomForestClassifier, and SHAP sample selection. Set at top of notebook via `np.random.seed(42)`.

**Key reproducibility notes:**
- SHAP computed on 300-row random sample of test set (`X_test.sample(300, random_state=42)`)
- SHAP output shape is `(300, 38, 2)` — use `shap_values[:, :, 1]` for positive class
- Precision@50 computed as `y_test.iloc[np.argsort(proba)[-50:]].mean()`
- All artifacts saved to `artifacts/` folder in working directory

**Expected outputs on re-run:**
- Precision@50: ~0.90 (RF) vs ~0.18 (baseline)
- impression_momentum SHAP rank: 1
- search_volume correlation: ~0.0012

---

## 9. Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset — [https://flyrank.ai](https://flyrank.ai)

Dataset: [FlyRank/internship-starter](https://huggingface.co/datasets/FlyRank/internship-starter) on Hugging Face.

Crediting the data source is standard research practice — it tells the world where this data came from.
