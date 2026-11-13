# Methodology

## Lane decision (provisional)
Refresh / Content Opportunity Scoring is a **candidate**. It is supported only if the real data has page-level daily clicks and impressions with at least about 120 days of history. If not, choose the best-supported lane and document why here. **Lane decision: PENDING schema inspection.**

## Assumptions
- Page-days absent from the warehouse mean no impressions (future clicks = 0). Verify against the real data.
- Equal-length windows make recent vs prior comparisons fair.

## Leakage controls
1. Features use rows dated on or before the decision date only (tested by changing future rows).
2. Label uses rows strictly after the decision date.
3. Validation pages are never in training (hash-based group split).
4. Training label window ends before the validation label window starts.
5. Features with near-perfect correlation to the label raise an error.

## Baseline
0.5 x percentile(falling clicks) + 0.3 x percentile(recent clicks) + 0.2 x percentile(recent impressions).

## Model
Gradient boosting (scikit-learn `HistGradientBoostingClassifier`), seed 42. Logistic regression is available as a comparison.

## Evaluation
ROC AUC, average precision, precision@10/50, lift@k, NDCG@k, plus a bootstrap interval for the AUC difference, all on the same validation rows.

## Honest framing
Observed = historical patterns. Directional = model rankings. Decision-support = review prompts. No causal claims; no claims about Google's algorithm.
