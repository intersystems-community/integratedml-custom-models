# AI Functions for IRIS IntegratedML

> SQL-native generative AI primitives running inside InterSystems IRIS via
> IntegratedML Custom Models.

## What this demo is

The InterSystems Developer Community article _"Introducing AI Functions:
Power of AI With the Simplicity of SQL"_ (originally a SingleStore announcement)
describes a set of AI primitives — `AI_COMPLETE`, `AI_SENTIMENT`,
`AI_TRANSLATE`, `EMBED_TEXT`, `AI_SUMMARIZE`, `AI_CLASSIFY`, `AI_EXTRACT` —
that are callable directly from SQL.

This demo recreates that capability **inside InterSystems IRIS** using the
IntegratedML _Custom Models_ feature. Each AI Function is implemented as a
self-contained `IRISModel` Python file that is loaded by IRIS's embedded
Python interpreter when you run `CREATE MODEL` / `TRAIN MODEL`. From SQL
the call site looks like this:

```sql
SELECT
    ticket_id,
    PREDICT(AISentiment USE message_text) AS sentiment,
    PREDICT(AIClassify  USE message_text) AS department,
    PREDICT(AISummarize USE message_text) AS summary
FROM AIFunctions.SupportTickets;
```

…instead of `aura.AI_SENTIMENT(...)` etc.

## What's in the box

| AI Function    | IRISModel file                | Default backend                      |
| -------------- | ----------------------------- | ------------------------------------ |
| `AI_SENTIMENT` | `iris_models/ai_sentiment.py` | VADER-style lexicon scorer           |
| `AI_CLASSIFY`  | `iris_models/ai_classify.py`  | TF-IDF + cosine similarity           |
| `AI_SUMMARIZE` | `iris_models/ai_summarize.py` | TextRank-lite extractive summarizer  |
| `AI_TRANSLATE` | `iris_models/ai_translate.py` | Phrase-book + word-level dictionary  |
| `EMBED_TEXT`   | `iris_models/embed_text.py`   | Hashing trick + L2-normalised TF-IDF |
| `AI_EXTRACT`   | `iris_models/ai_extract.py`   | Question-aware regex pipeline        |
| `AI_COMPLETE`  | `iris_models/ai_complete.py`  | Anthropic Claude (offline fallback)  |

The default backends are intentionally lightweight so the demo runs offline
with no API key. `AI_COMPLETE` switches to the real Claude API
(`claude-haiku-4-5` by default) when the `anthropic` Python package is
installed in the IRIS Python environment **and** `ANTHROPIC_API_KEY` is set.
You can swap any backend for a heavier model (HuggingFace, ONNX, etc.) by
editing the corresponding `IRISModel.predict()` — the SQL surface area stays
the same.

## Repo layout

```text
demos/ai_functions/
├── iris_models/         # The 7 self-contained IRISModel files (deployed to IRIS)
│   └── _staging/        # Per-function staging dirs (created by deploy_models.py)
├── sql/                 # Table setup + CREATE MODEL + use-case queries
├── data/                # CSV samples mirroring the article's tables
├── scripts/             # deploy_models.py
└── tests/               # pytest unit tests for each AI Function
```

## Quick start (offline, no IRIS required)

```bash
# Install demo deps (numpy, pandas, scikit-learn, pytest)
make install

# Run the unit tests
pytest demos/ai_functions/tests/ -v

# Run the end-to-end demo against the sample CSVs
python run_ai_functions_demo.py
```

`run_ai_functions_demo.py` reproduces the three article use cases — support
intelligence, real-time fraud detection, churn prediction — using the same
`IRISModel` classes that IRIS will load in production.

## Deploy into IRIS

1. Stage each AI Function into its own `pathtoclassifiers` directory:

   ```bash
   python demos/ai_functions/scripts/deploy_models.py
   ```

   This creates `demos/ai_functions/iris_models/_staging/<function_name>/`
   with one `.py` file each so each `CREATE MODEL` resolves unambiguously.

2. Mount the demo into the IRIS container (already done in
   `docker-compose.yml` via `./demos:/opt/irisapp/demos:ro`).

3. Run the SQL files in order:

   ```sql
   -- 1. tables
   /usr/irissys/bin/iris session iris -U USER < demos/ai_functions/sql/01_setup_tables.sql

   -- 2. (load CSV data via your tool of choice — see scripts/ in other demos)

   -- 3. register the models
   /usr/irissys/bin/iris session iris -U USER < demos/ai_functions/sql/02_create_models.sql

   -- 4. exercise the use cases
   /usr/irissys/bin/iris session iris -U USER < demos/ai_functions/sql/03_use_case_support_intelligence.sql
   /usr/irissys/bin/iris session iris -U USER < demos/ai_functions/sql/04_use_case_fraud_detection.sql
   /usr/irissys/bin/iris session iris -U USER < demos/ai_functions/sql/05_use_case_churn_prediction.sql
   ```

## Why one staging directory per AI Function?

IRIS's IntegratedML scans every `.py` under `pathtoclassifiers` and
instantiates every `IRISModel` class it finds. If we dropped all seven AI
Functions into a single directory, every `CREATE MODEL` would load all seven
and pick one essentially at random. `scripts/deploy_models.py` mirrors each
function into a dedicated subdirectory so each `CREATE MODEL` resolves to
exactly one model.

## Using a real LLM for `AI_COMPLETE`

To enable real Claude calls inside IRIS:

```bash
# Install the SDK into the IRIS Python environment.
docker exec integratedml_iris pip3 install anthropic

# Pass the API key via the IRIS process environment.
docker exec -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY integratedml_iris ...
```

Then re-train `AIComplete`. The model will detect the env var + SDK and call
the real Messages API; without them it falls back to the deterministic
offline stub (which is clearly labelled `[offline-stub]` in its output).
