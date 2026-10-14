# FinSight sentiment annotation guide

Owner: Nathan. Sprint 1 prepares A2/A7 and the W1 evidence handoff.

## What to label

Label a statement's expressed financial sentiment toward the row's target company.
This is sentiment in the evidence, not your view of the stock or a trading decision.
Read the linked filing section for context, especially when the excerpt says "we",
names a competitor, or describes a hypothetical risk. Keep the original text and
source fields unchanged. If the passage is wrong or too short, add a corrected
passage to the input JSON; a new ID will preserve the old annotation.

| Label | Meaning |
| --- | --- |
| positive | Favorable performance, demand, outlook, or financial conditions for the target. |
| negative | Adverse performance, constraints, or explicitly described financial risks for the target. |
| neutral | Relevant factual description without a clear favorable/adverse direction. |
| uncertain | Mixed signals, unclear attribution, or missing context prevents a defensible label. |
| not_applicable | No relevant statement about the target company's financial situation. |

Do not infer positivity from words such as "revenue" alone. Distinguish observed
results from future or conditional claims. When competitor performance is
described, avoid assuming its impact on the target unless the evidence says so.

## Reason categories

Enter zero or more IDs separated by semicolons in `gold_reasons`.

| ID | Covers |
| --- | --- |
| revenue | Sales, demand, revenue growth or decline. |
| profitability | Gross/net/operating margin, costs, profitability. |
| earnings | Profit/loss, EPS, earnings performance. |
| guidance | Management expectations, forecasts, outlook. |
| product_business_growth | Products, launches, adoption, expansion. |
| market_risk | Macroeconomy, supply chain, financing, broad market uncertainty. |
| regulation | Export controls, regulation, legal concerns. |
| competition | Competitive pressure, rivals, market positioning. |
| investor_analyst_expectations | Explicit investor/analyst expectations or reactions. |

Reasons describe topics supported by the statement. They are not automatic
explanations of the model's prediction. If context supports a label but no listed
reason fits, leave reasons blank and explain in `annotation_notes`.

## Workflow

1. Run `notebooks/p2_analysis_nlp.ipynb` to create/update the queue.
2. Fill `gold_sentiment`, `gold_reasons`, `annotator`, and `annotation_notes`.
3. Set `review_status` to `labeled`; ask a teammate to review a subset.
4. Set `review_status` to `reviewed` only after review and resolving disagreements.
5. Rerun the summary cell. `uncertain` and `not_applicable` are retained but excluded
   from the usable three-class evaluation count.

The supplied three real NVIDIA excerpts are unlabeled starters. Synthetic NER
fixtures never enter this queue. Expand to approximately 200 real statements from
the agreed company list, spread across periods and sources. Review the company
list with P1: the alias dictionary is seeded from Anna's notebook, not the app's
currently enabled companies. SEC-only coverage should be disclosed as a limitation.

Before tuning, reserve a held-out group of filings and deduplicate across splits.
Do not score on training examples or use model predictions as human ground truth.
For future PhraseBank/FinBERT work, record the exact model/dataset version,
attribution and license; see the [FinBERT model card](https://huggingface.co/ProsusAI/finbert).

## Evidence contract

An `N…` evidence note cites `S…` statement IDs. Each statement identifies its
`P…` passage, containing source URL, accession, form, period, and section. Entity
and sentence offsets are character offsets in that supplied passage, not filing
HTML or PDF page offsets. The result validates span/text consistency and rejects
notes with missing sources. These checks do not prove that a generated claim is
semantically supported; that remains W3 work.
