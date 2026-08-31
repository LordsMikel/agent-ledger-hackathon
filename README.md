# Agent Ledger

**Author:** Miguel Medina Cantos

**Date:** 2026-08-28

Agent Ledger is a platform that facilitates the extraction, embedding, and semantic search of
invoice data using Gemini 3.5 Flash and Vertex AI.

## Live Demo

Try the deployed Agent Ledger application at:

[https://agent-ledger-148369048204.europe-west1.run.app](https://agent-ledger-148369048204.europe-west1.run.app)


## Competition Track

**The Collaborative Partner** — Agent Ledger provides multi-turn dialogue through Google ADK and
real-time retrieval-augmented generation over persistent invoice knowledge in Cloud Firestore. It
combines agentic routing, semantic retrieval, deterministic financial analysis, and grounded
Gemini explanations to help users investigate invoices conversationally.



## Services Used and Required by the Competition

### Architecture Overview

- **Model:** Gemini 3.5 **Flash** on Vertex AI, selected as one of Google's most cost-efficient and
  capable multimodal models for invoice image extraction.
- **Embedding model:**
  [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  for semantic search and 384-dimensional vector indexing.
- **Dataset:**
  [`FATURA2`](https://huggingface.co/datasets/mathieu1256/FATURA2-invoices), a publicly available
  dataset of invoice images and structured invoice data.
- **Invoice storage:** Google Cloud Storage buckets store the initial invoice batch inputs. Each
  Vertex AI input JSONL contains the corresponding invoice image encoded in base64, together with
  its extraction request.

### Infrastructure Overview

#### Cloud Infrastructure

- **Vertex AI** runs Gemini 3.5 Flash for multimodal invoice extraction through sequential batch
  jobs.
- **Cloud Firestore** stores structured invoice data and MiniLM vector embeddings and provides
  vector similarity search for the RAG system.
- **Cloud Storage** stores the initial invoice batch inputs and the resulting Vertex AI prediction
  outputs during ingestion. The chat runtime does not read the bucket because its indexed invoice
  data is served permanently from Firestore.

#### Code Infrastructure

- **Google ADK (Agent Development Kit)** builds the agents that interact with Gemini and the
  Firestore-backed retrieval tools.
- **Domain-Driven Design (DDD)** organizes the system into explicit preparation, application,
  agent, embedding, infrastructure, and chat boundaries.
- **Loguru** records batch timings, processed and failed invoices, and remaining workload.

Google ADK agent definitions separate extraction, retrieval, presentation, and supervisor
responsibilities.

The architecture diagram is in [`docs/architecture.md`](docs/architecture.md).

## Application layout

```text
agent-ledger/
├── adk_invoice_chat/               # ADK Web discovery package
│   ├── __init__.py
│   └── agent.py                  # Exposes root_agent
├── config/
│   ├── __init__.py
│   └── settings.py
├── convert_batch/                 # Step 1: images -> JSONL -> Cloud Storage
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── index_batch_to_firestore/      # Step 2: Vertex Batch -> MiniLM -> Firestore
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── docs/
│   └── architecture.md
├── input/                         # 112 operational invoice JPEGs (9.99 MiB)
├── output/
│   ├── extracted/                 # Normalized extraction records
│   ├── logs/                      # Persistent Loguru timing streams
│   └── vertex_batches/            # Local JSONL and handoff manifest
├── manual_firestore_chat.py       # Real additional-image integration test
├── run_adk.py                     # Step 3: chat-only ADK Web launcher
├── Dockerfile                     # Chat-only ADK Cloud Run image
├── cloud-run.env.yaml             # Non-secret Cloud Run runtime settings
├── requirements-cloud-run.txt     # Minimal deployed chat dependencies
├── src/
│   ├── main.py
│   ├── application/
│   │   └── orchestrator.py
│   ├── embeddings/
│   │   ├── vector_search.py
│   │   └── vector_index.py
│   └── agents/
│       ├── extractor/agent.py
│       ├── search/agent.py
│       ├── chat/
│       │   ├── agent.py
│       │   ├── models.py
│       │   ├── prompts.py
│       │   └── service.py
│       └── orchestrator_agent.py
├── tests/
├── .env                            # Local configuration, excluded from Git
└── pyproject.toml
```

## Install

Use a project-local environment; the repository-level preparation environment is not required or
modified.

```bash
cd agent-ledger
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Configure Vertex AI, Cloud Storage, and Firestore in the project-local `.env` file:

```env
GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
FIRESTORE_DATABASE=agent-test1-100
GEMINI_BATCH_GCS_URI=gs://test-100-invoices/vertex-batches
INSERT_INTO_INDEX=false
```

`Settings.from_env()` loads this file automatically. Exported variables retain precedence for
normal settings, while the Gemini backend itself is always forced to Vertex AI. `.env` is excluded
from Git.

Gemini on Vertex AI and Firestore both use Google Cloud Application Default Credentials:

```bash
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com firestore.googleapis.com storage.googleapis.com \
  --project=<PROJECT_ID>
```

Create a private Cloud Storage bucket in the Google Cloud project before running batch ingestion,
grant Vertex AI permission to read and write its objects, and set `GEMINI_BATCH_GCS_URI` to the
desired prefix inside that bucket.

The application enforces Vertex AI at runtime. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are ignored.

## Validate the prepared input

```bash
.venv/bin/python -m src.main validate
.venv/bin/python -m src.main validate --cloud
```

The application always reads from the 112-image operational dataset in `agent-ledger/input/`, never
from the repository-level 8,600-image source dataset or the Parquet preparation artifacts.

## Create the Firestore vector index

Print the exact command configured for the current project, collection, field, database, and
384-dimensional MiniLM vectors:

```bash
.venv/bin/python -m src.main index-command
```

Review and execute the printed `gcloud firestore indexes composite create` command. Wait until the
index reports `READY` before searching. The index uses Firestore's required flat vector
configuration. Embeddings are unit normalized, so the default `DOT_PRODUCT` metric follows the
Firestore recommendation; set `VECTOR_DISTANCE_MEASURE=COSINE` for the normalization-safe
alternative.

## Complete 112-invoice batch workflow

The operational workflow has three independent entry points. This keeps Cloud Storage as an
ingestion transport only and prevents opening ADK Web from ever starting image processing.

### Step 1: prepare and upload the Vertex input

```bash
.venv/bin/python -m convert_batch.main
```

`convert_batch` accepts only the exact 112 images in the project-local `input/` directory. It
validates and hashes them, sorts them by filename, and produces 12 JSONL inputs: eleven groups of
ten and one final group of two. Every group is also guarded by a 10 MiB raw-image ceiling. The files
are written under `output/vertex_batches/<dataset-id>/inputs/` and uploaded to
`gs://test-100-invoices/vertex-batches/<dataset-id>/inputs/`.

The resulting `output/vertex_batches/manifest.json` is the immutable handoff to step 2. The dataset
ID depends on image hashes, extraction prompt, and Gemini model, so unchanged input has the same
identity. Existing GCS inputs are reused by default; use `--force` only to overwrite them. For a
local preparation check without any bucket write, use `--local-only`.

Loguru reports each prepared batch, its image count and size, elapsed time, upload/reuse status,
remaining images, and remaining batches. Logs persist under `output/logs/`.

### Step 2: run Vertex Batch and index Firestore

Keep ingestion disabled in `.env` and enable it only for this command:

```bash
INSERT_INTO_INDEX=true .venv/bin/python -m index_batch_to_firestore.main
```

The indexer is pinned to `FIRESTORE_DATABASE=agent-test1-100`. Before submitting a Vertex job, it
checks every deterministic Firestore document ID, image SHA-256, extraction model, and embedding
model. If all 112 records are current, it submits zero Vertex jobs, does not load MiniLM, and makes
no Firestore writes. On a partial rerun, complete batches are skipped and only missing or stale
records are embedded and written. A prepared ten-image Vertex input may be resubmitted when one
record in that group needs retrying, but current Firestore records are never rewritten.

The 12 Vertex jobs run sequentially. Gemini writes its prediction JSONL to the configured bucket;
the script maps results back to source images, saves normalized extraction JSON locally, generates
384-dimensional MiniLM vectors, and upserts Firestore. Loguru records job submission and state,
Vertex output parsing, MiniLM time, Firestore time, processed/failed counts, remaining images, and
remaining batches. If one item fails, successful records from that batch remain saved and the next
normal run retries only records that are not current. Use `--force` only for a deliberate complete
rebuild.

`INSERT_INTO_INDEX` remains false by default and the indexer checks it defensively before creating
any Vertex, MiniLM, or Firestore client. The preparation and chat commands do not override it. The
combined ingestion/ADK launcher has been removed; these three entry points are the complete
operational workflow.

### Step 3: launch the chat-only ADK interface

```bash
.venv/bin/python run_adk.py --port 8000
```

Open `http://127.0.0.1:8000` and select `adk_invoice_chat`. This launcher never reads images,
manifests, or Vertex output and never accesses the batch bucket. Its only runtime data source is
the permanent Firestore RAG collection through `aggregate_invoices`, `analyze_invoices`,
`list_all_invoices`, or `semantic_search_invoices`.

Create the 384-dimensional vector index for `agent-test1-100` separately and wait for `READY`.
Ingestion and `list_all_invoices` can work while it is building, but
`semantic_search_invoices` requires the ready vector index.

## Deploy the public ADK demo to Cloud Run

The root `Dockerfile` packages only `config/`, `src/`, and `adk_invoice_chat/`. MiniLM is downloaded
during the image build and then forced into offline mode. The build contexts explicitly exclude
`.env`, invoice images, extracted output, local ADK state, tests, and ingestion scripts. Cloud Run
starts the official ADK API server with its demonstration UI; it does not execute `run_adk.py`.

Create a dedicated runtime service account once and grant only Vertex AI execution and read-only
Firestore access:

```bash
export PROJECT_ID="<PROJECT_ID>"
export CLOUD_RUN_REGION="europe-west1"
export CLOUD_RUN_SERVICE="agent-ledger"
export RUNTIME_SA_NAME="agent-ledger-cloud-run"
export RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  --project="${PROJECT_ID}"

gcloud iam service-accounts create "${RUNTIME_SA_NAME}" \
  --project="${PROJECT_ID}" \
  --display-name="Agent Ledger Cloud Run"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/datastore.viewer"
```

Build from the repository root and publish the competition UI:

```bash
gcloud run deploy "${CLOUD_RUN_SERVICE}" \
  --source=. \
  --project="${PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --service-account="${RUNTIME_SA}" \
  --env-vars-file=cloud-run.env.yaml \
  --port=8080 \
  --cpu=4 \
  --memory=4Gi \
  --concurrency=4 \
  --timeout=3600 \
  --min-instances=0 \
  --max-instances=1 \
  --cpu-boost \
  --execution-environment=gen2 \
  --allow-unauthenticated
```

The public setting is intentional for the competition demo. The runtime identity cannot write or
delete Firestore records, ingestion is disabled twice in configuration and code, and the maximum
instance count limits unexpected Gemini traffic. The ADK UI is a demonstration interface rather
than a production frontend.

Read the resulting URL and verify that ADK exposes only the invoice application:

```bash
export APP_URL="$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${CLOUD_RUN_REGION}" \
  --format='value(status.url)')"

curl --fail --silent --show-error "${APP_URL}/list-apps"
```

The Cloud Run container uses in-memory ADK sessions and a single maximum instance for this demo.
Conversation state can be lost after scale-to-zero or a revision replacement; indexed invoice data
remains permanently stored in Firestore.

## Search invoices

```bash
.venv/bin/python -m src.main search "invoices from Acme in March" --limit 5
```

The query is embedded with the same MiniLM model and normalization settings as the indexed text.
Results include the Firestore-calculated vector distance, source image path, and extracted invoice
fields. Firestore returns an empty JSON list when no indexed documents match.

## Chat with the agentic RAG

The chat agent delegates invoice questions once to the `search_invoices` retrieval agent through an
ADK `AgentTool`. That agent exposes four mutually exclusive traceable child branches. Targeted or
semantic questions use MiniLM plus `semantic_search_invoices`. Unfiltered collection inspection uses
`list_all_invoices`. Supplier-total questions use `aggregate_invoices`, while other structured
filters, grouping, statistics, thresholds, sorting, and user-requested evidence limits use
`analyze_invoices`.

A concrete invoice-number detail request always keeps the original semantic RAG path. The search
result includes the complete stored OCR, allowing Gemini to present general information, parties,
line items, totals, bank details, payment terms, and the source image. Such a request never uses the
generic analytical count/table response.

### Agentic routing with deterministic analytics

```text
                 USER QUESTION
                       |
                       v
                Gemini Agent
                       |
          +------------+-------------+-------------+
          |            |             |             |
          v            v             v             v
 supplier totals   structured     full-list      semantic
                   analytics      inspection      detail
          |            |             |             |
          v            v             v             v
 aggregate_       analyze_        list_all_      semantic_search_
 invoices         invoices        invoices       invoices
          |            |             |             |
          |       Python metrics     +---- Gemini --+
          |            |                  grounding
       Python      Gemini explains              |
    final answer    precomputed facts            |
          |            |                         |
          |       Python appends                 |
          |       invoice tables                 |
          +------------+------------+------------+
                                   v
                              Final answer
```

This split prevents global questions from aggregating incomplete semantic subsets. Both analytical
tools read the same existing Firestore documents and use exact Python `Decimal` arithmetic; they do
not change the database, create an index, or trigger reindexing. Python also renders their Markdown
evidence tables. `aggregate_invoices` remains a completely direct deterministic response.
`analyze_invoices` gives a specialized Gemini child only compact precomputed facts for a concise
explanation, then Python appends any invoice rows. Gemini does not calculate figures or regenerate
long invoice tables.

`analyze_invoices` provides a controlled SQL-like contract without SQL: structured filters;
`group_by` over supplier, currency, date, year, or month; `count`, `sum`, `average`, `minimum`, or
`maximum`; a HAVING-like condition; sorting; group limits; dynamic invoice evidence counts; and
optional evidence. Requested quantities are copied from the user rather than hardcoded. For example,
"30 invoices from the same month and year" uses year/month grouping, `count >= 30`, and an evidence
count of 30. "15 invoices from a specific date" uses an exact date filter and an evidence count of
15. When no group meets a threshold, the tool returns the closest result and every tied group.

Explicit period comparisons are also bounded deterministically. `Compare 2002 with 2009` becomes a
single `year in 2002,2009` filter, separated by currency. Python precomputes totals, counts,
averages, extrema, absolute differences, and percentage changes. Gemini explains the comparison,
and Python appends the invoice tables for 2002 and 2009 only; unrelated years are never included.

Run one chat turn with:

```bash
.venv/bin/python -m src.main chat "Which Acme invoices do I have from March?" \
  --user-id miguel \
  --session-id finance-session
```

The JSON response contains the answer, model, session, tools used, finish reason, configured output
ceiling, and actual token usage. The default generation ceiling is 8,000 output tokens and can be
changed with `CHAT_MAX_OUTPUT_TOKENS`.

```json
{
  "max_output_tokens": 8000,
  "token_usage": {
    "prompt_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_tokens": 0,
    "thought_tokens": 0,
    "tool_prompt_tokens": 0
  },
  "tools_used": ["search_invoices", "analyze_invoices"]
}
```

`CHAT_SEARCH_LIMIT` bounds targeted vector results and defaults to 10. `CHAT_LIST_LIMIT` bounds a
global metadata read and defaults to 200, enough for the prepared 112-image dataset. If that limit
is reached, the tool marks its response as truncated and discloses that aggregation scope. Global
calculations keep currencies separate and exclude missing or invalid amounts.

Sessions are managed through ADK's in-memory session service for now. Reusing a user ID and session
ID preserves context only while the same application process remains alive; durable chat sessions
are a future infrastructure concern.

## Manual additional-image Firestore and chat test

The manual integration script skips the first three sorted images already indexed by the initial
smoke test. It submits images `000003.jpg` through `000022.jpg` as one strict twenty-item Gemini
Batch API job, writes their embeddings and invoice data to Firestore, and then opens a continuous
terminal chat backed by the same Firestore RAG pipeline:

```bash
.venv/bin/python manual_firestore_chat.py --user-id miguel --session-id smoke-test
```

The script enables ingestion, skips the first three images, and fixes the batch size to twenty only
for its own process; it does not change `.env` or the normal 112-image limit. Existing current
documents are skipped. Pass `--force` to repeat only these twenty extra images. Type `salir`,
`exit`, or `quit` to close the chat. Every answer displays the tools used and Gemini token usage.
This test calls billable cloud services and requires the Firestore vector index to be `READY` before
the chat can search it.

To use Google's official ADK development interface after the same ingestion, run:

```bash
.venv/bin/python manual_firestore_chat.py --interface adk-web --port 8000
```

Open `http://127.0.0.1:8000`, select `adk_invoice_chat`, and chat with `invoice_chat`. The discovery
package exposes the required `root_agent`; its search tool still embeds each question with MiniLM
and retrieves invoice evidence from Firestore. ADK Web is a local development and debugging UI, not
a production frontend.

## Google ADK agents

`src/agents/orchestrator_agent.py` composes three specialized `google.adk.agents.Agent` instances:

- `invoice_extractor` for grounded multimodal extraction;
- `invoice_search` for Firestore RAG retrieval;
- `invoice_chat` as the primary RAG agent, with a nested `search_invoices` AgentTool;
- `search_invoices` exposes `aggregate_invoices` for deterministic supplier totals,
  `analyze_invoices` for generic deterministic analytics, `list_all_invoices` for unfiltered
  inspection, and `semantic_search_invoices` for targeted retrieval;

The ADK supervisor delegates between these agents. Deterministic indexing remains in the
application orchestrator so retries, duplicate detection, batching, and financial values do not
depend on LLM routing behavior.

## Tests

The tests use deterministic fake extractors, embeddings, and vector storage. They do not download
the MiniLM model, call Gemini, or require Firestore credentials.

```bash
.venv/bin/python -m pytest
# Or with only the standard-library runner:
.venv/bin/python -m unittest discover -s tests -v
```

Cloud integration still requires a configured Google Cloud project, enabled APIs, Application
Default Credentials, and a ready Firestore vector index. Those external resources are deliberately
not created or mutated by the test suite.
