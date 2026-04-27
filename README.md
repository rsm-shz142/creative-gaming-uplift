# Creative Gaming: Uplift Modeling for Ad Campaign Targeting

Causal uplift modeling framework for Creative Gaming's Zalon ad campaign —
identifying which customers are genuinely persuaded by ads rather than
likely buyers, to maximize incremental profit.

## Background

Creative Gaming needed to decide which of 120,000 customers to target with
ads for their new game Zalon ($14.99 per purchase, $1.50 ad cost per customer).
Standard propensity models waste budget on "sure things" — customers who would
buy regardless of the ad. This project uses uplift modeling to estimate the
true causal effect of advertising per customer.

The breakeven uplift threshold: $1.50 / $14.99 ≈ 0.10 — only target customers
whose incremental purchase probability exceeds 10%.

## Methodology

Creative Gaming's random ad allocation policy created natural RCT data:
- **Control group:** 30,000 customers with no ad
- **Treatment group:** 30,000 randomly selected customers who received the ad

For each model, two separate classifiers were trained (treatment and control),
and the uplift score was computed as the difference in predicted probabilities:
Uplift = P(purchase | ad=1, X) − P(purchase | ad=0, X)

Four model types were built and compared:

| Model | Hyperparameters Tuned |
|---|---|
| Logistic Regression | Baseline |
| Neural Network (MLP) | Hidden layers, alpha, learning rate |
| Random Forest | n_estimators, max_depth, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, min_child_weight |

## Key Results

- Uplift targeting **consistently outperformed propensity targeting** across all
  four models when targeting the best 30,000 of 120,000 customers
- Optimal targeting under uplift model (~15% of customers) was far more
  concentrated than propensity model (~50%), reducing wasted ad spend
- Top 20% of customers by uplift score showed >20% incremental purchase
  probability; bottom decile showed **negative uplift** (do-not-disturb segment)
- Correlation analysis confirmed uplift score negatively correlated with
  baseline purchase probability (ρ ≈ −0.66), correctly de-prioritizing
  "sure things"

## Files

- `creative_gaming_uplift.ipynb`: Full analysis notebook
- `data/`: RCT dataset files (control, treatment, and random groups)

## Tools

Python · Polars · pyrsm · scikit-learn · XGBoost ·
Logistic Regression · Neural Network · Random Forest ·
Uplift Modeling · RCT · Cross-Validation · AUC
