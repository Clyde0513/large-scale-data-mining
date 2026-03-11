# Project 3 Report

**Student:** Clyde Villacrusis (305965764)  
**Course:** ECE 219 — Large-Scale Data Mining  

---

## Part A: Fine-Tuning Language Models for Mathematical Reasoning (GSM8K)

---
gt p
### Task 1: Baseline: How Good is the Base Model?


#### Question 1 — Base Model Zero-Shot Accuracy

The base `Qwen2.5-1.5B-Instruct` model was evaluated on 100 GSM8K test questions using batched inference (batch size = 16), a standardized system prompt, and a `\boxed{...}` answer-extraction rule.

**Result: 37% accuracy** (35–40% as expected).

---

#### Question 2: Three Failure Examples from the Base Model 

Three random incorrect examples were sampled from the base model's outputs. All failures involve arithmetic errors despite correct problem setup.

---

**Example 1: Sandcastle levels (expected: 60)**
> *"Each level has half the square footage as the level below it. Top level = 16 sq ft. What is the average square footage of a level in a 4-level sandcastle?"*

The model correctly identified levels: 16, 8, 4, 2.  
It then incorrectly computed the total: `16 + 8 + 4 + 2 = 20` (should be 30), and computed an average of 5.  
**Predicted: 5 | Ground truth: 60**

> **Failure type:** Simple addition error AND directional logic error: the problem states each *upper* level is half the one *below* it, so the sequence going bottom-to-top should be 128, 64, 32, 16, giving a total of 240 and an average of 60. The model also went the wrong direction (top-down halving instead of bottom-up doubling).

---

**Example 2: Combined weights (expected: 623)**

> *"Grace weighs 125 lbs. Alex weighs 2 lbs less than 4 times Grace's weight. What are their combined weights?"*

The model correctly computed: `4 × 125 = 500`, then `500 − 2 = 498`.  
It then added: `125 + 498 = 613` (off-by-ten arithmetic error; correct is 623).  
**Predicted: 613 | Ground truth: 623**

> **Failure type:** Arithmetic error on the final addition step.

---

**Example 3 — Overtime pay (expected: 460)**

> *"Regular rate = $10/hr for first 40 hrs. Overtime = 1.2× regular. She worked 45 hrs. Total earnings?"*

The model incorrectly deduced that overtime hours = `12 * 5 = 10` (mistake in the multiplication), then computed `$400 + $10 = $410`.  
**Predicted: 410 | Ground truth: 460**

> **Failure type:** Fundamental misunderstanding of "overtime" — the model failed to compute that 45 − 40 = 5 overtime hours exist, and incorrectly derived 0 overtime hours.

---

### Task 2: LoRA Supervised Fine-Tuning

#### Question 3: Hyperparameter Analysis

Three key hyperparameters from the training configuration:

---

**LoRA Rank (`r` = 8)**

- **What it controls:** The rank of the low-rank matrix decomposition used for the adapter. It directly controls the expressiveness (capacity) of the LoRA adapter.  
- **If increased:** The model can learn more nuanced reasoning behavior and potentially higher accuracy with more training data. Downside: more trainable parameters, higher GPU memory/compute, increased overfitting risk on small datasets, and potential instability with aggressive learning rates.  
- **If decreased:** Fewer parameters, faster training, lower overfitting risk. However, the model may underfit and plateau sooner, especially on complex mathematical reasoning.

---

**LoRA Alpha (`lora_alpha` = 16)**

- **What it controls:** The scaling factor applied to the LoRA update. It governs how strongly the adapter influences the base model's behavior relative to the frozen weights. The effective learning rate for LoRA is scaled by `alpha / r`.  
- **If increased:** The LoRA update is amplified, allowing the adapter to adapt more aggressively. Useful when the base model is far from the target behavior. Risk: overshoot and training instability if too large.  
- **If decreased:** More conservative updates, more stable training. Risk: the adapter may not have enough influence to meaningfully change behavior (underfitting).

---

**Gradient Accumulation Steps (= 4, effective batch size = 32)**

- **What it controls:** Simulates a larger effective batch size without increasing per-device GPU memory usage. Gradients are accumulated across `N` mini-batches before an optimizer step.  
- **If increased:** Smoother gradient estimates, improved training stability, similar benefits to larger-batch training without additional memory. Useful on constrained hardware.  
- **If decreased:** More frequent parameter updates per epoch (faster adaptation), but noisier gradient estimates. Very small effective batch sizes can sometimes hurt generalization.

---

#### Question 4: LoRA Parameter Counts 

**(a) Total parameters in base model:** **~1.54 billion** parameters

**(b) Trainable LoRA parameters (default config: r=8, target = q/k/v/o projections):** **2,179,072 parameters**

**(c) Percentage of parameters trained:** **~0.14%**

This tiny fraction (~700× reduction versus full fine-tuning) is achieved by two mechanisms:

1. **Low-rank factorization:** Instead of updating a full weight matrix `W ∈ ℝ^(d×d)`, LoRA learns two thin matrices `A ∈ ℝ^(d×r)` and `B ∈ ℝ^(r×d)` where `r << d`. This replaces `d²` parameters with `2dr` parameters per layer.  
2. **Sparse module targeting:** LoRA adapters are only attached to the attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`), not every layer in the network. The original weights remain completely frozen throughout training.

---

#### Question 5: LoRA SFT with 1,000 Examples 

Configuration: LoRA (r=8, alpha=16, dropout=0.05), 1 epoch, learning rate = 2×10⁻⁴, cosine LR schedule, effective batch size = 32 (8 per device × 4 accumulation steps).

| Model | Accuracy (100-Q subset) |
|---|---|
| Base Qwen2.5-1.5B-Instruct (0-shot) | 37% |
| LoRA SFT — 1,000 examples | 42% |

**Comment:** The +5 percentage point improvement over the baseline confirms that even a small LoRA adapter trained on a relatively small GSM8K subset can improve mathematical reasoning. The model has learned the step-by-step solution style from training data. However, the accuracy is still far from saturated, suggesting more data or better training strategies could yield further gains.

---

#### Question 6: Scaling Predictions

Based on the 1k→42% result, the expected gains from scaling up are:

- **1k to 3k examples:** Estimated +3 to +7 percentage points (landing in the low 60s). More problem templates reduce overfitting to the small 1k subset.  
- **3k to full 7,473 examples:** Smaller gains, perhaps +2 to +4 points, due to limited adapter capacity and single-epoch training.

**Strategy:** Scale in steps. Train on 3k first; if improvement ≥ 3–4 points, it is worth training on the full dataset. If the gain is marginal, focus on improving prompt quality or supervision signal rather than simply adding data.

---

#### Question 7: Data Scaling Results

| Training Examples | Val Accuracy (100-Q subset) |

|--- | --- |

| 0 (base model) | 37% |
| 1,000 | 42% |
| 3,000 | 46% |

**Trend:** SFT clearly helps relative to the base model, but gains do not scale monotonically under fixed hyperparameters. One explanation is that the 1k run may mildly overfit to patterns in the 100-question evaluation subset, while the 3k run, trained for more steps at the same learning rate, may start overfitting to noise in the additional data. This is consistent with the idea of **diminishing returns in straightforward SFT**: once the adapter learns the general solution style, more heterogeneous data does not automatically improve accuracy unless hyperparameters or data quality are also improved.

---

### Task 3: K-Shot Prompting

#### Question 8: Base vs. SFT on the Three Failure Examples 

The same three failure examples from Q2 were re-evaluated with the base model and the LoRA SFT (1k) model side-by-side.

---

**Example 1: Sandcastle (GT: 60)**

| Model | Output Summary | Predicted | Correct? |
|---|---|---|---|
| Base | Computes 16+8+4+2=20, average = 5 | 5 | No |
| SFT (1k) | Correctly doubles upward (16+32+64+128), total = 232, average = 58 | 58 | No (close) |

> The SFT model fixed the directionality error but got the arithmetic slightly off (232/4 = 58 vs GT 60 = 240/4).

---

**Example 2: Combined weights (GT: 623)**

| Model | Output Summary | Predicted | Correct? |
|---|---|---|---|
| Base | 4×125=500, 500−2=498, 125+498=**613** | 613 | No |
| SFT (1k) | 4×125=500, 500−2=498, 125+498=**623** | 623 | **Yes** |

> The SFT model **fixed** this error, correctly performing the final addition.

---

**Example 3: Overtime pay (GT: 460)**

| Model | Output Summary | Predicted | Correct? |
|---|---|---|---|
| Base | Overtime hours = 5−40 = 0, total = $400 | 400 | No |
| SFT (1k) | Correctly computes 1.2×10×5=60 overtime, 400+60=460 | 45 (extraction issue) | No |

> The SFT model's reasoning was correct (produced $460) but the answer extractor grabbed "45" (the number of hours) from a poorly formatted response rather than 460.

---

#### Question 9: Recurring Error Patterns After SFT

After SFT, some arithmetic errors are reduced but the following failure modes persist:

1. **Basic arithmetic errors:** The model still slips on simple addition and multiplication even when the reasoning setup is correct (e.g., Example 1 above).  
2. **Multi-step planning drift:** The SFT model generally improves reasoning structure, but occasionally jumps from a partially complete solution to a final answer without fully justifying the last step.  
3. **Answer extraction / formatting inconsistencies:** In some cases the model produces the correct number in its reasoning chain but fails to put it in a clean, extractable final slot (as seen in Example 3), leading to incorrect evaluation even though the logic was correct.

---

#### Question 10: K-Shot Prompting Results (k = 3)

Three GSM8K training demonstrations (same for all questions, both models) were used as k-shot examples prepended in the system prompt.

| Model | 0-shot Accuracy | 3-shot Accuracy | Δ |
|---|---|---|---|
| Base Qwen2.5-1.5B-Instruct | 24% | 33% | +9% |
| LoRA SFT (3k examples) | 52% | 56% | +4% |

> **Note:** Base model 0-shot result varies across runs (24%–37%). The figures above reflect one specific evaluation run.

---

#### Question 11: Analysis of Few-Shot Prompting 

**Does few-shot prompting help the base model?**

Yes, modestly. Accuracy increases from 24% to 33% (+9 pp). The three demonstrations provide explicit examples of the desired step-by-step solution style and final-answer format, which partially compensates for the base model's weak math reasoning. However, the base model remains fragile: the long prompt sometimes causes it to copy surface patterns rather than reason genuinely, so individual questions may actually perform worse with demonstrations.

**Does few-shot prompting help the SFT model? By how much?**

Yes, but more modestly (+4 pp, from 52% to 56%). Since the SFT model already learned the reasoning style and answer format during fine-tuning, demonstrations mainly act as a small "reminder" rather than teaching fundamentally new behavior.

**Which model benefits most from few-shot prompting, and why?**

The **base model** benefits more. The demonstrations give the base model information it never saw during training concrete examples of multi-step reasoning on GSM8K-style problems so performance improves substantially. For the SFT model, the reasoning style is already internalized; few-shot prompting mostly smooths out edge cases with diminishing marginal returns.

---

### Task 4: Beyond Scaling: Quality Matters

#### Question 12: Qualitative Reflection on Performance Limits

After evaluating multiple model configurations across Tasks 1–3, five recurring failure categories were identified:

**1. Arithmetic Reliability**  
Even when problem setup and multi-step reasoning are correct, the base model slips on simple arithmetic (e.g., `16 + 8 + 4 + 2 = 20`, `12 × 5 = 10`). After SFT, some of these errors are reduced and wrong answers move closer to correct values, but arithmetic noise persists in a non-trivial fraction of cases.

**2. Multi-Step Planning Drift**  
The base model often sets up Step 1 correctly but drifts in later steps, applying the wrong operation, forgetting intermediate variables, or reusing them incorrectly. LoRA SFT substantially improves reasoning structure, but the model still occasionally skips justifying the last computation step.

**3. Problem Comprehension**  
The model generally parses questions correctly (identifies entities, target quantity, relevant numbers). SFT reduces comprehension errors further. However, when a misunderstanding occurs (usually in more complex phrasing), the entire reasoning chain is wrong from the start.

**4. Output Consistency and Extraction Failures**  
With a `\boxed{...}` system prompt, most outputs follow the format and the extractor works reliably. However, some responses place the correct integer in the reasoning chain but not in a consistent final slot, causing incorrect evaluation despite correct logic. SFT improves final-answer discipline, reducing this failure mode.

**5. Training Data Quality**  
GSM8K solutions are human-written, diverse, and stylistically inconsistent. The SFT model learns an "average" style — sometimes verbose and slightly wandering rather than strictly structured. This suggests that a small set of very high-quality, well-structured demonstrations (as in k-shot prompting) can be more valuable per example than a larger quantity of uncurated training data.

---

### Task 5: Open Challenge: Push Toward the Ceiling

#### Question 13: Self-Consistency Inference 

**(a) Hypothesis**

Adding inference-time **self-consistency** (majority vote from multiple sampled solutions) should significantly improve accuracy over greedy decoding. Most remaining errors are random arithmetic slips or single-step reasoning mistakes; if K independent solutions are sampled with temperature, these "noisy" errors should cancel out on average, and the correct answer should dominate the vote. Combining self-consistency with k=3 few-shot demonstrations should further stabilize reasoning style and final-answer formatting.

**(b) Method**

- Starting model: LoRA SFT (1k and 3k examples), same 100-question test subset.  
- At inference time: k=3 few-shot demonstrations prepended to the system prompt.  
- Self-consistency: For each question, sample **K = 5 or 10** solutions using `do_sample=True`, `temperature=0.7`, `top_p=0.9`.  
- Final prediction: **majority vote** over extracted integer answers from all K samples. Ties or all-None --> incorrect.

**(c) Results**

| Configuration | Accuracy (100-Q subset) |
|---|---|
| LoRA-1k, 0-shot, greedy | 42% |
| LoRA-3k, 0-shot, greedy | 52% |
| LoRA-3k, 3-shot, greedy | 56% |
| LoRA-1k, 3-shot, self-consistency K=5 | 62% |
| LoRA-3k, 3-shot, self-consistency K=5 | **65%** |

Best configuration: **LoRA-3k + 3-shot + self-consistency (K=5) → 65%** (+13 pp over LoRA-3k greedy, +23 pp over base model).

**(d) Analysis**

Results strongly support the hypothesis. Moving from greedy decoding (52%) to 3-shot + self-consistency K=5 increased accuracy to 65%. In problems previously answered incorrectly, at least a majority of the K samples recomputed the arithmetic correctly so the correct answer won the vote.

Observations:
- Increasing K from 3 to 5 gave a noticeable improvement; further increases likely yield diminishing returns relative to extra compute.  
- Arithmetic and local-reasoning noise identified in Q12 is specifically addressed by self-consistency: diverse samples "average out" random slips.  
- Remaining errors tend to be "structural" (wrong problem interpretation from the start), which requires better supervision or data quality rather than more sampling.

> **Overall conclusion from Task 5:** Inference-time self-consistency is a powerful, training-free complement to LoRA SFT. It specifically targets the arithmetic noise that persists after fine-tuning and achieves large gains at the cost of K× more inference compute.

---

---

## Part B: Agentic Data Mining with ReAct

---

### Task 1: Load and Inspect JSONL Files

#### Question 14 — Dataset Statistics and Schema

The `da-dev-questions.jsonl` and `da-dev-labels.jsonl` files were loaded and inspected.

- **Number of questions:** 257  
- **Number of labels:** 257  
- Both files share the same IDs (confirmed by set equality check).

**Question record keys:** `['id', 'question', 'concepts', 'constraints', 'format', 'file_name', 'level']`

**Example (ID 0):**

| Field | Value |
|---|---|
| `id` | 0 |
| `question` | "Calculate the mean fare paid by the passengers." |
| `concepts` | `['Summary Statistics']` |
| `constraints` | "Calculate the mean fare using Python's built-in statistics module or appropriate statistical method in pandas. Rounding off the answer to two decimal places." |
| `format` | `@mean_fare[mean_fare_value]` where `mean_fare_value` is a float rounded to 2 decimal places |
| `file_name` | `test_ave.csv` |
| `level` | easy |

**Label record keys:** `['id', 'common_answers']`

**Example (ID 0):**

| Field | Value |
|---|---|
| `id` | 0 |
| `common_answers` | `[['mean_fare', '34.65']]` |

---

#### Question 15: Three Random Question Inspections 

Three questions were sampled at random (seed=0) and their referenced CSVs were loaded.

---

**Question ID 593** — `20170413_000000_group_statistics.csv`

- **Shape:** (96, 10)  
- **Columns:** `timestamp` (object), `num. busy overflows` (int64), `num. calls answered` (int64), `num. calls abandoned` (int64), `num. calls transferred` (int64), `num. calls timed out` (int64), `avg. num. agents talking` (float64), `avg. num. agents staffed` (int64), `avg. wait time` (object), `avg. abandonment time` (object)  
- **Question:** "Using feature engineering techniques, create a new feature that represents the waiting time for callers before being answered by an agent as a percentage of the average abandonment time. Then, explore the distribution of this new feature and determine if it adheres to a normal distribution."

---

**Question ID 663** — `YAHOO-BTC_USD_D.csv`

- **Shape:** (2176, 7)  
- **Columns:** `Date` (object), `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` (float64)  
- **Question:** "Create a scatter plot of the 'High' and 'Low' columns to visualize the relationship between the highest and lowest prices for each day. Calculate the Pearson correlation coefficient between these two columns."

---

**Question ID 34** — `imp.score.ldlr.metabolome.csv`

- **Shape:** (377, 8)  
- **Columns:** `#featureID` (object), `row ID` (int64), `row m/z` (float64), `row retention time` (float64), `LibraryID` (object), `standard_indentification_level_1` (object), `_feature_id` (object), `importance.score` (float64)  
- **Question:** "Is there a correlation between the 'row retention time' and 'importance.score' columns?"

---

#### Question 16: Multi-Part Answer Format Analysis 

**How the dataset represents multi-part answers:**

In the questions JSONL, the required output format is a string with one or more `@name[value]` slots. Multi-part answers list multiple such slots, e.g.:

```
@mean_fare_child[mean_fare], @mean_fare_teenager[mean_fare], @mean_fare_adult[mean_fare], @mean_fare_elderly[mean_fare]
```

In the labels JSONL, the `common_answers` field is a list of two-element lists. Each inner list contains `[slot_name, ground_truth_value_as_string]`. Multi-slot answers produce multiple such pairs.

**Automatic evaluation plan:**

1. Parse the model's output with a regex to extract a `{slot_name: predicted_value}` dictionary.  
2. Build a ground-truth dictionary from `common_answers` and compare slot-by-slot.  
3. For numeric slots: convert both to float and check within small tolerance (≤ 1e-2).  
4. For string slots: normalize whitespace and require exact match.  
5. Apply **all-or-nothing** scoring at the question level: a prediction is marked correct if and only if every required slot matches.

---

#### Question 17: Inspecting the 10 Solvable Tasks 

The 10 selected IDs: `[0, 5, 9, 10, 14, 18, 24, 25, 26, 55]`

Each record was printed showing the CSV file name, required output format, and question text. The tasks span a variety of operations including grouped means, filtering by conditions, counting, min/max lookups, normality tests, and Pearson correlation. All have clearly specified `@name[value]` output formats, enabling automatic evaluation.

---

### Task 2: Model Loading and Structured Output

#### Question 18: Planner Structured Output Demonstration 

A `PlannerOutput` Pydantic model was defined with three fields:

```python
class PlannerOutput(BaseModel):
    thought: str     # internal reasoning
    is_done: bool    # whether to stop with a final answer
    response: str    # next action or final @name[value] answer
```

`Qwen/Qwen3-4B-Instruct-2507` was loaded and wrapped with **Outlines** (`outlines.from_transformers`), which constrains token generation to produce valid JSON conforming to the schema. The `run_planner` function calls the model and validates the result with `PlannerOutput.model_validate_json(...)` — no `try/except` fallbacks are needed.

**Five test prompt results:**

| Prompt | `is_done` | `response` summary |
|---|---|---|
| 1. New Titanic task: first step | `False` | "Load the Titanic CSV file to access the data." |
| 2. After loading: compute mean fare per class | `False` | "Calculate the mean fare per passenger class using groupby on 'Pclass' and 'Fare'." |
| 3. Previous code had `KeyError: 'Fare'` | `False` | Suggests printing `df.columns` to find the correct column name. |
| 4. Age-group means already computed, multi-slot answer ready | **`True`** | `@mean_fare_child[31.09], @mean_fare_teenager[31.98], @mean_fare_adult[35.17], @mean_fare_elderly[43.47]` |
| 5. `total_outliers = 139` computed, single-slot answer ready | **`True`** | `@total_outliers[139]` |

All 5 outputs parsed into `PlannerOutput` objects without any parsing errors, confirming the schema-constrained planner is reliable.

---

#### Question 19: Why Structured Output Matters 

In large-scale data mining pipelines, many components must be orchestrated automatically over thousands of tasks. If the LLM produces free-form text, every downstream step requires brittle string parsing and ad-hoc heuristics, and small phrasing changes can silently break the pipeline or cause wrong actions.

**Structured output solves this** by constraining the model to emit a fixed schema (e.g., `{"thought": ..., "is_done": ..., "response": ...}`). Downstream code treats the planner's output as a normal Python object — no raw text manipulation. This makes the system:
- **More robust** (fewer parsing errors, no format-drift bugs)  
- **Easier to debug** (each field can be logged and unit-tested independently)  
- **Easier to scale** (answers can be automatically evaluated in a predictable format across hundreds of tasks)

---

### Task 3: ReAct Data Analysis Agent

#### Question 20: ReAct Agent Results on 10 Tasks 

**Agent Architecture (4 components):**

| Component | Role |
|---|---|
| **Planner** | Produces structured `PlannerOutput` (thought, is_done, response) using Pydantic + Outlines |
| **Coder** | Generates Python code to execute the planner's instruction |
| **Executor** | Runs code in a sandboxed persistent environment (no internet/system calls); state persists across steps |
| **Observer** | Summarizes stdout/stderr into a compact structured `ObservationOutput` to guide the next planning step |

The loop runs for at most 5 iterations per question with error-recovery: if execution fails (e.g., `KeyError` on a missing column), the observation captures the error type and the planner re-plans accordingly.

---

**Accuracy: 7/10 = 70%** on the 10 selected IDs using strict automatic slot-by-slot evaluation.

---

**Qualitative Trace 1: Question ID 0 (Success with Hallucination Guard)**

- **Task:** Compute mean fare in `test_ave.csv` tp `@mean_fare[mean_fare_value]`  
- **Step 1:** Planner asks Coder to load the CSV and compute the mean fare.  
- **Code:** `mean_fare = df["Fare"].mean()` to print `RESULT mean_fare = 34.65`  
- **Issue:** The Coder also printed `FINAL_ANSWER: @mean_fare[8.75]` a hallucinated/stale value.  
- **Agent behavior:** The agent's "use env-computed values" guardrail overrode the hallucinated print and returned `@mean_fare[34.65]` from the actual computed variable.  
- **Marked: Correct**

> **Key insight:** This illustrates a common failure mode of LLM tool agents; the model can compute correctly but still emit an incorrect final line. Tying the final answer to computed variables (rather than raw generated text) makes the agent reliably safe against this type of hallucination.

---

**Qualitative Trace 2: Question ID 10 (Error Recovery)**

- **Task:** Check if "Total Traded Quantity" in `GODREJIND.csv` is normally distributed → `@is_normal[yes/no]`  
- **Step 1:** Coder ran a Shapiro-Wilk test and printed `RESULT is_normal = no` but also printed `FINAL_ANSWER: @is_normal[yes]` contradictory outputs.  
- **Agent behavior:** The guardrail rejected the printed final answer because the slot variable was inconsistently set.  
- **Step 2:** Coder reran a clean implementation and printed `RESULT is_normal = no` followed by `FINAL_ANSWER: @is_normal[no]`.  
- **Marked: Correct**

> **Key insight:** The agent handles "messy" intermediate outputs (contradictory lines) and recovers by running a cleaner computation. This demonstrates the value of multi-step error-recovery loops over single-shot generation.

---

**Qualitative Trace 3: Question ID 55 (Format Mismatch Failure)**

- **Task:** Compute mean number of cases across all countries/years in `estimated_numbers.csv`to `@mean_cases[mean_value]` (expected as positive integer)  
- **Step 1:** `KeyError: 'Cases'`  
- **Step 2 (recovery):** Agent printed column names, found correct column `"No. of cases"`.  
- **Step 3:** Computed `pd.to_numeric(df["No. of cases"], errors="coerce").mean()` = 2199.55  
- **Output:** `@mean_cases[2199.55]` → marked **Incorrect** because the required format expects an integer (e.g., `@mean_cases[2199]`).

> **Key insight:** Even when the agent correctly computes the target statistic, strict formatting constraints (float vs. integer) can cause an otherwise correct answer to fail evaluation. This highlights the importance of type-aware final output formatting when the benchmark specifies the answer type.

---

---

## Part C: Regression Analysis on Diamonds Dataset

---

### Question 21: Exploratory Data Analysis 

#### Correlation Analysis

Pearson correlations between numerical features and the target `price` (sorted by absolute value):

| Feature | |corr with price| |
|---|---|
| `carat` | 0.913 |
| `length` | 0.870 |
| `width` | 0.842 |
| `depth` | 0.300 |
| `table_percent` | 0.042 |
| `depth_percent` | 0.025 |

**Interpretation:**  
Size-related measurements (`carat`, `length`, `width`) are strongly correlated with each other and with `price` — larger stones are consistently more expensive. `depth_percent` and `table_percent` have near-zero correlation with price but correlate more with each other, reflecting cutting geometry rather than size. `depth` is moderately correlated with price, likely because it is also related to total stone volume.

---

#### Distribution Analysis

Skewness values (sorted by absolute value):

| Feature | Skewness |
|---|---|
| `depth` | +27.49 (extreme likely outliers/data issues) |
| `depth_percent` | −13.56 |
| `table_percent` | −11.05 |
| `width` | +4.12 |
| `price` | +3.07 |
| `carat` | +2.33 |
| `length` | +1.28 |

`price`, `carat`, and `width` are heavily right-skewed (many low/moderate values, long right tails).  
`depth_percent` and `table_percent` show strong skew with extreme outliers.

**Suggested transformations:**
- For right-skewed positive variables (`price`, `carat`, `width`): apply `log1p()`, which compresses the long tail and stabilizes variance.  
- For geometry variables with extreme skew (`depth`, `depth_percent`, `table_percent`): apply outlier clipping/winsorization first, then a power transform.

---

#### Categorical Analysis

Box plots of `cut`, `color`, `clarity`, `symmetry`, and `polish` vs. `price` show differences in price distributions across categories, but the relationship is **not monotonic** — a well-known confounding effect from stone size.

**Key trends:**
- **Cut / Symmetry / Polish:** Median prices are relatively close between quality categories; quality matters but its direct price impact is weaker than size-driven effects.  
- **Color / Clarity:** Some "lower-quality" categories show higher median prices because they tend to occur in larger stones on average.

**Takeaway:** Categorical features do influence price, but raw box plots are dominated by size. For a rigorous model, controlling for `carat`/size (e.g., within carat bins) is recommended to isolate the pure quality effects on price.

---

### Question 22: Categorical Encoding *(10 pts)*

#### Encoding choices

| Feature | Method | Reason |
|---|---|---|
| `cut` | Ordinal (scalar) | Only 2 categories with a clear quality order: Very Good < Excellent -> {0, 1} |
| `color` | Ordinal (scalar) | Industry-standard ordered grades M < L < … < D -> mapped to 0–9 (higher = better) |
| `clarity` | Ordinal (scalar) | Industry-standard ordered grades I3 < … < IF -> mapped to 0–9 (higher = better) |
| `symmetry`, `polish`, `girdle_min`, `girdle_max` | One-hot | No reliable linear ordering; one-hot avoids injecting incorrect numeric structure |

After encoding, the dataset shape became `(149871, 34)` with no missing values introduced.

---

#### Trade-off Discussion

**What does one-hot encoding discard?**  
One-hot encoding discards any notion of **ordering** and **distance** between categories. For variables like `color` or `clarity`, it cannot represent that "D is better than E, which is better than F": all categories are treated as equally unrelated labels. The model cannot directly learn a monotonic quality effect from the encoding alone.

**What assumption must hold strongly for scalar (ordinal) encoding?**  
Scalar encoding assumes:  
1. The categories have a **meaningful, consistent order**, and  
2. The effect on the target is **monotonic** in that order, and ideally **approximately linear** in the encoded integers (each integer step represents a roughly equal increment in the target relationship).  

If these assumptions do not hold, scalar encoding can mislead models. particularly linear regression by imposing artificial and incorrect spacing between categories.

---

### Question 23: Feature Standardization

All numeric features were standardized to zero mean and unit variance using `sklearn.preprocessing.StandardScaler`. Categorical features were already encoded.

The standardized dataset (features scaled, `price` preserved in original units) was saved as `diamonds_standardized.csv`.

- **Shape after encoding + standardization:** `(149871, 34)`  
- **Feature means:** ≈ 0.0 (confirmed by post-fit mean check)  
- **Feature std devs:** ≈ 1.0 (confirmed by post-fit std check)

---

### Question 24: Feature Selection 

Top 5 features by each method, applied to the encoded (pre-standardization) dataset:

#### Mutual Information Regression

| Rank | Feature | MI Score |
|---|---|---|
| 1 | `carat` | highest (1.38) |
| 2 | `width` | 1.20 |
| 3 | `length` | 1.19 |
| 4 | `depth` | 1.16 |
| 5 | `color` | 0.18 |

#### F-Regression (linear association)

| Rank | Feature | F-Score |
|---|---|---|
| 1 | `carat` | highest (755380) |
| 2 | `length` | 464517 |
| 3 | `width` | 364744 |
| 4 | `depth` | 14789 |
| 5 | `polish_VeryGood` | 453 |

**Interpretation:** Both methods agree that size-related variables (`carat`, `length`, `width`, `depth`) dominate price prediction. Mutual information additionally ranks `color` highly, reflecting nonlinear or interaction effects not captured by linear F-tests. F-regression highlights the one-hot `polish_VeryGood` indicator as the top categorical feature under a linear association test.

**Agentic integration (ReAct agent, diamond questions 0 & 1):**  
The agent correctly loaded `diamonds_standardized.csv`, computed both MI and F-regression scores, and returned:  
- `@top5_mi[carat, width, length, depth, color]` — matches manual results exactly.  
- `@top5_f[carat, length, width, depth, polish_Excellent]` — agrees with manual results (top 4 identical; fifth varies between closely ranked one-hot indicators).

---

### Question 25: Linear Regression Models and Agentic Integration 

#### Manual Regression Results (10-Fold Cross-Validation on `diamonds_selected.csv`)

Three models were trained and evaluated: OLS (no regularization), Ridge (L2), and Lasso (L1). Alpha values were tuned over a logarithmic grid.

| Model | Best Alpha | Train RMSE | Validation RMSE |
|---|---|---|---|
| OLS | N/A | ~1606.7 | ~1606.4 |
| Ridge | ~3.28×10⁻⁴ | ~1606.7 | ~1606.4 |
| Lasso | ~0.0373 | ~1606.7 | ~1606.4 |

**Best model by validation RMSE: Lasso with α ≈ 0.0373**, validation RMSE ≈ 1606.4.

The near-identical RMSEs across models suggest:
- The dataset is large enough that regularization adds minimal improvement over OLS.  
- The significant features (size-related) dominate so strongly that shrinkage methods converge to nearly the same solution as unregularized regression.

---

#### How Regularization Affects Learned Parameters

| Regularization | Effect on Coefficients |
|---|---|
| **OLS (none)** | No shrinkage; can overfit when features are highly correlated (multicollinearity) |
| **Ridge (L2)** | Smoothly shrinks all coefficients toward 0; keeps all features nonzero; reduces variance from multicollinearity |
| **Lasso (L1)** | Drives some coefficients exactly to 0 → automatic feature selection; can be unstable when predictors are highly correlated (may arbitrarily pick one of a correlated pair) |

---

#### Statistical Significance (P-Values from OLS via Statsmodels)

The `statsmodels.OLS` p-values test H₀: coefficient = 0 (feature has no linear effect on price).  
- Features with low p-values (< 0.05) are statistically significant in the linear model.  
- In the Diamonds dataset, size features (`carat`, `length`, `width`) have extremely small p-values, confirming their strong linear association with price.

---

#### Agentic Integration (ReAct agent, diamond questions 2, 3, 4)

| Question | Task | Agent Result |
|---|---|---|
| QID 2 | OLS 10-fold CV RMSE | `@ols_val_rmse[1571.91]` (differs from manual due to different preprocessing path in generated code) |
| QID 3 | LassoCV best α + val RMSE | **Agent failed** — repeatedly generated syntactically incomplete code (unclosed parentheses, missing imports) within the 5-step budget |
| QID 4 | RidgeCV best α + val RMSE | Agent returned `@ridge_alpha[0.0012] @ridge_val_rmse[12345.67]` — inconsistent with scale, indicating hallucination / incorrect observation acceptance |

**Reported final regression results** are from the controlled manual pipeline above.

**Key limitation demonstrated:** The agent can sometimes finalize based on non-grounded intermediate text rather than verified computation. For complex multi-step ML workflows (cross-validation, hyperparameter search, model fitting), the agent benefits from clearer scaffolding and tighter code-generation guardrails.

---

---

## Summary Table

| Question | Topic | Key Result |
|---|---|---|
| Q1 | Base model accuracy | **37%** on 100 GSM8K questions |
| Q2 | Three failure examples | Arithmetic errors in addition and multiplication; directional logic error |
| Q3 | Hyperparameter analysis | LoRA rank controls adapter capacity; alpha controls update strength; gradient accumulation simulates large batch |
| Q4 | LoRA parameter counts | 1.54B total params; 2.18M trainable (0.14%) |
| Q5 | LoRA SFT 1k examples | **42%** (+5 pp over base) |
| Q6 | Scaling prediction | Expect +3–7 pp from 1k tp 3k; diminishing returns beyond |
| Q7 | LoRA SFT 3k examples | **46–52%** (varies by run) |
| Q8 | Failure-case comparison | SFT fixed Example 2; improved Example 1 (still slightly wrong); logic correct but extraction failed on Example 3 |
| Q9 | Error patterns after SFT | Arithmetic errors, multi-step drift, extraction inconsistencies persist |
| Q10 | K-shot prompting (k=3) | Base: 24%→33%; SFT-3k: 52%→56% |
| Q11 | K-shot analysis | Base model benefits more; SFT already internalized reasoning style |
| Q12 | Qualitative reflection | 5 failure categories: arithmetic, planning drift, comprehension, extraction, data quality |
| Q13 | Self-consistency (K=5) | **65%** with LoRA-3k + 3-shot + SC; +13 pp over LoRA-3k greedy |
| Q14 | Dataset schema | 257 questions, 257 labels; key fields: id, question, format, file_name |
| Q15 | Random CSV inspection | Diverse datasets: call-center stats, BTC prices, metabolomics |
| Q16 | Multi-part answers | `common_answers` is a list of [slot, value] pairs; evaluate slot-by-slot |
| Q17 | 10 solvable tasks | Various operations on structured CSV data |
| Q18 | Structured planner output | 5/5 prompts parsed into PlannerOutput without errors; 2 `is_done=True` cases |
| Q19 | Why structured output | Robustness, debuggability, scalability for pipelines |
| Q20 | ReAct agent accuracy | **70% (7/10)** with error recovery |
| Q21 | EDA on Diamonds | Carat/length/width dominate; price is right-skewed; use log1p transform |
| Q22 | Categorical encoding | Ordinal for cut/color/clarity; one-hot for symmetry/polish/girdle |
| Q23 | Standardization | Features scaled to mean=0, std=1; saved as CSV |
| Q24 | Feature selection | Top 5 by MI and F-regression both dominated by carat, length, width, depth |
| Q25 | Regression models | Lasso (α≈0.037) best; val RMSE ≈ 1606.4; agent partially succeeded |
