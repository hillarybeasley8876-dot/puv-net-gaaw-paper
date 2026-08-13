# Fresh Reviewer Trace

- Reviewer: GPT-5.6-Sol, ultra
- Review class: same-family, provisional
- Mode: read-only adversarial experiment-integrity audit
- Verdict: FAIL

The fresh reviewer independently read the evaluation, dataset, metric, configuration, result and manuscript files. Its decisive findings were:

1. PU1K ground truth is loaded from the dataset and is not generated from model output.
2. Training uses a fixed permutation for validation, whereas the diagnostic scripts sampled the array tail. The archived 200-row slice contains 11 held-out and 189 training records.
3. The formal metric path independently normalizes prediction and ground truth, while the post-hoc diagnostic path uses a shared ground-truth coordinate frame. `evaluate.py` also records metadata inconsistent with the squared-distance implementation.
4. D1 requires `scale_qk=true`, but `measure_cv_nn.py` reconstructed the default `scale_qk=false` model. State-dict compatibility did not expose the error.
5. Archived row-level JSON values recompute, but the current evidence is a post-hoc patch diagnostic rather than the formal 127-model PU1K test evaluation.
6. One seed per configuration and an 11-record post-hoc intersection cannot support statistical significance, robustness, generalization or SOTA claims.

The reviewer required withdrawal of the old 200-validation-sample, 6.43%-plus-CD/HD-improvement and D1 claims. The corrected paper now reports the 11-record intersection only, identifies CD/HD as shared-ground-truth-frame diagnostics, excludes D1, and keeps the overall experiment-audit verdict visible as FAIL/provisional.
