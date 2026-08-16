# Ranking-Based Ensemble Methodology: Binary to Multiclass Adaptation Analysis

## 1. Current Methodology Overview

### Core Concept
The binary classification ranking methodology uses **probabilistic ensemble voting** combined with **positional ranking metrics** to identify and rank samples by their likelihood of belonging to the positive class.

### Key Functions

#### 1.1 `calculate_probabilities_for_all_models()`
- Each trained ML model produces probability scores: `model.predict_proba()[:, 1]`
- For binary classification: only class 1 probability is kept
- Creates a dataframe where each column contains probabilities from one model

#### 1.2 Ranking Process
```
For each model:
  1. Sort records by model probability (descending)
  2. Apply secondary sort by col_year, col_diameter (ascending)
  3. Assign rank = 1 to N
```

#### 1.3 Ranking Metrics (Positional Metrics)
Each metric normalizes the ranking position relative to ground truth labels (col_target = 1):

- **`calculate_top_accuracy()`**: Captures top-k precision
  - Measures % of target records found in top X% of ranked records
  - Weighted by importance of different ranking tiers

- **`calculate_median_accuracy()`**: Measures median rank position
  - Compares median rank of actual targets vs ideal targets
  - Normalized score: actual_median_score / ideal_median_score × 100

- **`calculate_power_accuracy()`**: Exponential distance weighting
  - Position weight = (100 - 100×(rank-1)/N)³ / 10000
  - Heavily penalizes targets ranked near end of list
  - Mean of position weights for target records

- **`calculate_ndcg_accuracy()`**: Normalized Discounted Cumulative Gain
  - Information retrieval metric: DCG = rel[0] + Σ(rel[i] / log₂(i))
  - Top-k variant: nDCG@k = actual_DCG / ideal_DCG × 100

#### 1.4 Ensemble Aggregation

**Method 1: Average Rank**
- Remove highest and lowest rank for each record across models
- Average remaining ranks
- Resort by average rank

**Method 2: Median Rank**
- Compute median rank across all models for each record
- Resort by median rank

**Method 3: Voting Ensemble**
- VotingClassifier with soft voting (average probabilities)
- Soft voting: probability = mean(model_probabilities)
- Create ranking from ensemble probability

### 1.5 Best Strategy Selection
```python
# Select best ensemble method based on metric scores
selected_rows = metrics_results.loc['Voting', 'Average', 'Median']
best_method = selected_rows.sort_values(by=score_sort_list, ascending=False).head(1)
# score_sort_list typically ranks by: ['power', 'nDCG', 'median', ...]
```

---

## 2. Adaptation for Multiclass Classification

### 2.1 ✅ What CAN Be Directly Applied

1. **Positional Ranking Metrics** (Top-k Accuracy, Median, Power, nDCG)
   - These metrics only depend on ranking positions, not binary labels
   - Adaptation: Change from `col_target == 1` to `col_target == target_class`
   - Can create separate rankings for each class vs. "all other classes"

2. **Ensemble Aggregation Methods** (Average Rank, Median Rank)
   - Averaging and median operations work on any set of ranks
   - No modification needed here

3. **Voting Classifier Ensemble**
   - VotingClassifier in scikit-learn supports multiclass automatically
   - `soft` voting computes: probability = mean(model_probabilities) across all models
   - Returns (N_samples, N_classes) probability matrix

4. **Overall Framework**
   - Model-specific ranking → Ensemble aggregation → Metric evaluation
   - This structure is classification-agnostic

---

### 2.2 ❌ What NEEDS Modification

#### Challenge 1: One-vs-Rest Ranking Philosophy
**Current Binary Approach:**
- Single ranking: positive vs. negative class
- Each record has ONE rank position

**Multiclass Extension (Two Options):**

**Option A: One-vs-Rest (OvR) Approach** ✅ Minimal changes
```
For each class (k = 1 to N_classes):
    For each model (m = 1 to N_models):
        prob_class_k = model_m.predict_proba()[:, k]
        rank = sort_by_prob_descending(prob_class_k)

    Average_rank_k = average(rank_across_models)
    Median_rank_k = median(rank_across_models)

    Calculate metrics with col_target == k
```
- **Result:** Each class has its own ranking
- **Use case:** "Which records are most likely to be Class A?", "Which are most likely to be Class B?"
- **Metrics:** Can still use all power/nDCG/median metrics per class

**Option B: Probability-of-Most-Likely-Class Ranking** ✅ Moderate changes
```
For each model (m = 1 to N_models):
    max_prob = max(predict_proba()) across all classes
    class_pred = argmax(predict_proba())
    rank = sort_by_max_prob_descending()

Average_rank = average(rank_across_models)
Median_rank = median(rank_across_models)

Calculate metrics based on class_pred vs col_target
```
- **Result:** Single ranking by confidence of prediction
- **Use case:** Rank by overall prediction confidence
- **Limitation:** Doesn't explicitly rank by specific class likelihood

#### Challenge 2: Metrics with No Ground Truth Class
**Binary:** Every record either has target=1 or target=0 (binary)
**Multiclass:** Can have records with target=1, 2, 3, etc.

**Solution:**
```python
# For Class 1 performance:
target_records = df[df[col_target] == 1]
non_target_records = df[df[col_target] != 1]

# Apply original metrics to OvR ranking
top_accuracy_for_class_1 = calculate_top_accuracy(
    df_ranked_by_class_1_prob,
    col_target=1,  # Changed to compare against class 1
    ...
)
```

#### Challenge 3: VotingClassifier Probability Output
**Binary:** predict_proba() → (N, 2) array
**Multiclass:** predict_proba() → (N, K) array

**Conversion for ranking:**
```python
# Binary (current):
prob_positive = model.predict_proba()[:, 1]  # Shape: (N,)

# Multiclass:
probs_multiclass = model.predict_proba()  # Shape: (N, K)

# Option A - Rank by each class's probability:
for class_k in range(K):
    prob_class_k = probs_multiclass[:, class_k]  # Shape: (N,)
    rank_class_k = sort_and_rank(prob_class_k)

# Option B - Rank by max probability:
max_probs = probs_multiclass.max(axis=1)  # Shape: (N,)
rank_max = sort_and_rank(max_probs)
```

---

## 3. Recommended Implementation Strategy for Satellites Project

### 3.1 Prerequisites
Check your multiclass classification setup in `satellites`:
```python
# In your data:
n_classes = df[col_target].nunique()  # Should be > 2
# Example: col_target = [1, 2, 3, 4]  for 4-class problem
# Or: col_target = [0, 1, 2, 3]  for 4-class problem
```

### 3.2 Implementation: One-vs-Rest Multiclass Ranking

**Step 1: Adapt `calculate_probabilities_for_all_models()` for Multiclass**
```python
def calculate_probabilities_for_multiclass(df_features, df_predictions, models, n_classes):
    """
    For multiclass: creates n_classes ranking columns
    """
    df_predictions = df_predictions.copy()

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_features)  # Shape: (N, n_classes)
            for class_idx in range(n_classes):
                col_name = f"{model_name}_prob_class_{class_idx}"
                df_predictions[col_name] = probs[:, class_idx]

    return df_predictions
```

**Step 2: Create Per-Class Rankings**
```python
def create_multiclass_rankings(df, models, col_id, col_target, n_classes,
                               rank_col_prefix="rank"):
    """
    Creates separate rankings for each class
    """
    df_predictions = df.copy()

    for target_class in range(n_classes):
        # Get probability columns for this class
        prob_cols = [f"{m}_prob_class_{target_class}" for m in models.keys()]

        # Method 1: Average rank
        df_predictions[f'{rank_col_prefix}_avg_class_{target_class}'] = \
            df_predictions[prob_cols].rank(ascending=False).mean(axis=1)

        # Method 2: Median rank
        df_predictions[f'{rank_col_prefix}_median_class_{target_class}'] = \
            df_predictions[prob_cols].rank(ascending=False).median(axis=1)

    return df_predictions
```

**Step 3: Adapt Metrics for Per-Class Evaluation**
```python
def calculate_multiclass_metrics(df_ranked, col_target, n_classes,
                                  top_perc_dict, col_rank_prefix="rank"):
    """
    Evaluates ranking quality for each class
    """
    results = []

    for target_class in range(n_classes):
        # Get ranking column for this class
        col_rank = f"{col_rank_prefix}_median_class_{target_class}"

        # Filter to records of this class and others
        df_class = df_ranked.copy()
        df_class['is_target_class'] = (df_class[col_target] == target_class).astype(int)

        # Apply ranking metrics
        metrics = calculate_rank_metrics_for_each_model(
            df_class,
            model_name=f"Class_{target_class}",
            col_target='is_target_class',  # Binary: 1 if target class, 0 otherwise
            col_rank=col_rank,
            top_perc_dict=top_perc_dict,
            top_rank_perc=0.2
        )
        results.append(metrics)

    return pd.DataFrame(results)
```

**Step 4: Integrate into Satellites Pipeline**
```python
# In GeospatialClassificationPipeline or ml_classification_pipeline:

# Get multiclass probabilities
n_classes = len(df[target_column].unique())
df_predictions = calculate_probabilities_for_multiclass(
    df_features, df_predictions, trained_models, n_classes
)

# Create per-class rankings
df_ranked_multiclass = create_multiclass_rankings(
    df_predictions, trained_models, col_id, target_column, n_classes
)

# Evaluate rankings
multiclass_metrics = calculate_multiclass_metrics(
    df_ranked_multiclass, target_column, n_classes, top_perc_dict
)
```

---

## 4. Pros and Cons of Adaptation

### ✅ Advantages

1. **Interpretability**: For each class, you get a clear ranking of "most likely to be Class X"
2. **Reuses Proven Metrics**: Top-accuracy, Power, nDCG all have established interpretations
3. **Class-Specific Confidence**: Can identify uncertain samples (low probability for all classes)
4. **Ensemble Benefit**: Voting still averages predictions intelligently
5. **Scalable**: Works for any number of classes (binary → multiclass easily)

### ❌ Limitations

1. **Computational Overhead**: K times more rankings (one per class)
2. **Interpretation Complexity**: Multiple rankings to analyze instead of one
3. **Class Imbalance Issues**: Rare classes may have poor rankings due to fewer samples
4. **Memory**: Need to store K ranking columns instead of 1
5. **Metric Selection**: "Best method" selection becomes multidimensional (which class? which metric?)

---

## 5. Alternative Approaches (If OvR Ranking Not Suitable)

### Option 1: Confidence-Based Ranking
- Rank by **predicted probability of most likely class**
- Use multiclass accuracy/F1 metrics instead of ranking metrics
- Simpler, but loses per-class ranking information

### Option 2: Hierarchical Ranking
- Create binary problem: "Class A vs. All Others"
- Apply binary ranking methodology
- Repeat for each class of interest

### Option 3: Probability Entropy Ranking
- Rank by **prediction uncertainty**: entropy = -Σ(p_k × log(p_k))
- Low entropy = confident predictions (rank high)
- Useful for flagging uncertain samples

---

## 6. Summary & Recommendation

| Aspect | Binary Approach | Multiclass OvR Adaptation |
|--------|-----------------|---------------------------|
| Ranking Metrics | ✅ Direct use | ✅ Per-class application |
| Ensemble (Voting) | ✅ Works | ✅ Works (multiclass default) |
| Rank Aggregation | ✅ Average/Median | ✅ Average/Median per class |
| Complexity | Low | Medium |
| Interpretability | High (1 ranking) | Medium (K rankings) |
| **Recommendation** | N/A | **✅ Recommended for satellites** |

### Implementation Priority
1. **✅ Start with Option B (One-vs-Rest)**: Minimal code changes, proven framework
2. Test with your actual multiclass data (agricultural parcels)
3. Compare against standard multiclass metrics (weighted F1, macro F1)
4. If performance is good, integrate into `GeospatialClassificationPipeline`

### Key Files to Modify in Satellites
- `src/ml_classification/ml_classification_pipeline/steps/` - evaluation step
- `src/ml_classification/ml_classification_pipeline/` - add multiclass ranking functions
- `src/common_libraries/` - consider adding multiclass-specific ranking utilities
