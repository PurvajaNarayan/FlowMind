# data/

Runtime data — **not committed** (see `.gitignore`). Each teammate populates this locally.

## Layout

```
data/
├── train_full.json          1319 records  (code 261 / instruct 407 / wiki 651)
├── test_full.json            953 records  (no key overlap with train)
├── images/
│   ├── main/<key>.png        top-down layout (flowchart TD) — the default
│   └── bottom_top/<key>.png  same charts, bottom-top layout (directional-bias set)
└── raw/                      source-scrape CSVs for the instruct set (redundant w/ JSON)
```

Both image folders hold 2532 PNGs = all train + test keys. There is **no image field
in the JSON** — join a record to its image by key: `data/images/main/<key>.png`
(use `flowmind.data.image_path_for` / `iter_qa(..., layout=...)`).

## Record schema (spec §6)

All subsets share `key`, `mermaid`, `summary`, `qa`. Differences:

| Subset | prefix | has `code` | extra fields |
|--------|--------|-----------|--------------|
| code | `code…` | ✅ | — |
| instruct | `instruct…` | ❌ | `text`, `title`, `url`, `category1/2` |
| wiki | `wiki…` | ❌ | `text`, `title`, `tags`, `category1/2` |

QA `type` ∈ {`fact_retrieval`, `applied_scenario`, `flow_referential`, `topological`}.
`A1`/`A2`/`A3` are paraphrases of the same gold answer — score against all three.

## Two image layouts (VLM note)

`main` (TD) and `bottom_top` are the *same* charts rendered two ways — FlowVQA's
directional-bias robustness set. Train/eval the VLM on `main`; use `bottom_top` as a
robustness ablation ("does the model read the graph, or memorize layout position?").

## How to get it

FlowVQA: https://github.com/flowvqa/flowvqa — download the JSON + image sets into this
folder. Check the dataset's license/citation terms before using images in any public
write-up or demo (spec §12).
