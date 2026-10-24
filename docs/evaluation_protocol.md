# Evaluation Protocol

## Objective
Compare baseline RAG, RAG with verification, and RAG with verification plus bounded self-correction on a fixed dataset.

## Dataset
`data/evaluation/starter_dataset.json` is a small synthetic smoke-test dataset. It is not a benchmark and must not be used to claim general performance. Create separate development and held-out test sets before tuning thresholds.

## Experimental controls
- Use the same documents, questions, model configuration, generation settings, and initial top-k for all modes.
- Record software versions, random seeds where applicable, hardware, model ID, retrieval parameters, and timestamps.
- Include answerable, unanswerable, multi-passage, irrelevant-evidence, contradiction, misleading-premise, and numerical-claim examples.
- Include initially correct answers to measure correction regressions.
- Report all failed requests and missing measurements.

## Metrics
- Claim support rate = supported claims / claims with valid verification labels.
- Unsupported claim rate = insufficient or contradicted claims / claims with valid verification labels.
- Answer coverage = answered questions / all questions.
- Abstention rate = abstained questions / all questions.
- Mean latency = total measured query latency / all attempted questions.
- Correction success rate = initially failing eligible cases accepted as correct after correction / eligible initially failing cases.
- Correction regression rate = initially acceptable cases made unacceptable by correction / initially acceptable cases subjected to correction.

For defensible answer correctness and context precision/recall, use reference answers and annotated evidence. For contradiction precision/recall, use human-labelled claim-evidence pairs. Use confidence intervals and paired statistical tests when sample size and assumptions support them. Do not tune thresholds on the held-out test set.

## Current implementation caveats
The current evaluation script outputs prototype summaries and latency, and does not yet calculate validated answer correctness, RAGAS metrics, context precision/recall, calibration, confusion matrices, cost, correction success/regression, or statistical significance. These are research TODOs, not completed features.
