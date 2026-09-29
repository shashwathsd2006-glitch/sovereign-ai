# Sovereign AI: On-Premise Agentic AI Workbench

A private, air-gap-friendly AI workbench for industrial maintenance and inspection teams.
Everything — inference, embeddings, vector search, document storage — runs on your own machine.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
6. [Model Routing](#model-routing)
7. [AI Agent / Maintenance Assessment](#ai-agent--maintenance-assessment)
8. [Security and Privacy](#security-and-privacy)
9. [Project Structure](#project-structure)
10. [Prerequisites](#prerequisites)
11. [Installation](#installation)
12. [Running the Project Locally](#running-the-project-locally)
13. [Demo Workflow](#demo-workflow)
14. [Example Model Routing Demo](#example-model-routing-demo)
15. [Troubleshooting](#troubleshooting)
16. [GitHub Repository Notes](#github-repository-notes)
17. [Current Prototype Status](#current-prototype-status)
18. [Future Improvements](#future-improvements)
19. [Hackathon Context](#hackathon-context)
20. [License](#license)

---

## Overview

**Sovereign AI** is a full-stack, on-premise AI workbench that lets an industrial organisation ask
questions about its own equipment documentation and inspection photographs **without sending a single
byte to a cloud AI provider**.

### The problem it solves

Industrial organisations sit on decades of maintenance manuals, SOPs, inspection reports and
photographs — but that knowledge is hard to use in the field:

- Engineers waste time flipping through PDFs to find one torque specification.
- Inspection images are reviewed by eye, with no consistency between shifts.
- Cloud AI assistants cannot be used because manuals, photographs and process data are
  classified, and policies forbid uploading them to third-party model APIs.
- Off-the-shelf "AI for industry" products require a vendor cloud connection that never
  satisfies an audit.

### Why on-premise / private AI matters for industrial organisations

- **Data residency** — maintenance manuals, defect photographs and process data never leave the site.
- **No vendor lock-in or egress** — inference runs on hardware you already own.
- **Auditability** — answers carry document name and page citations, so a claim can be traced to
  its source rather than trusted blindly.
- **Availability** — the system keeps working when the plant is offline or the WAN link is down.
- **Compliance** — aligns with internal information-classification rules that block public LLMs.

Sovereign AI demonstrates that a complete agentic workflow (vision + retrieval + reasoning + routing)
can run entirely against a local [Ollama](https://ollama.com) server.

---

## Key Features

| Feature | Page | What it does |
| --- | --- | --- |
| **AI Assistant** | `/assistant` | Chat-style interface. Routes questions either to direct LLM generation or to the RAG pipeline and shows cited sources. |
| **Knowledge Base / RAG** | `/knowledge` | Ask natural-language questions against indexed documents; returns a grounded answer plus document/page citations with similarity scores. |
| **Document Management** | `/documents` | Upload (PDF only, ≤ 50 MB), index, list, inspect and delete documents. Extraction, chunking and embedding happen synchronously on upload. |
| **Image Analysis / Vision** | `/image-analysis` | Upload an inspection photograph (JPEG/PNG/WebP/GIF, ≤ 20 MB); LLaVA returns a structured JSON defect report: issue, severity, confidence, visual evidence, recommended action. |
| **AI Agents** | `/agents` | Multi-step maintenance assessment: vision analysis **+** document retrieval **+** combined reasoning. See [Pump P-102 example](#ai-agent--maintenance-assessment). |
| **Model Router** | `/model-router` | Classifies each request (image / document / plain text) and dispatches it to the correct pipeline. Returns the routing decision, reason, processing path and timing. |
| **Security Center** | `/security` | Live posture dashboard: on-premise mode, Ollama status, installed models, ChromaDB state, embedding availability, CORS origins, document/chunk counts. |
| **Activity Logs** | `/activity` | Timeline view of system events *(currently sample data — see [Current Prototype Status](#current-prototype-status))*. |
| **Local Ollama Inference** | — | `qwen3-coder:latest` for text, `llava:latest` for vision. No external model API. |
| **Source-grounded responses** | — | RAG answers are constrained to retrieved chunks and instructed to cite document name and page, or to say the information was not found. |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Browser  (http://localhost:3000)                                        │
│  Next.js 16 · React 19 · TypeScript · Tailwind 4 · Radix UI             │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  HTTP / JSON  (NEXT_PUBLIC_API_URL)
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI backend  (http://127.0.0.1:8000)                                │
│                                                                          │
│   /documents   /rag   /vision   /agents   /model-router   /security      │
│                              │                                           │
│                     ┌────────┴─────────┐                                 │
│                     │  Model Router    │  classify(request)              │
│                     └───┬────────┬─────┘                                 │
│          image ─────────┘        └──────────┬──────────────┐             │
│                                            │              │             │
│              ┌─────────────────────────────┘              │             │
│              ▼                                            ▼             │
│   ┌────────────────────┐                    ┌──────────────────────┐     │
│   │ Vision path        │                    │ Text paths           │     │
│   │ LLaVA              │                    │ RAG + Qwen  /  Qwen  │     │
│   └─────────┬──────────┘                    └──────────┬───────────┘     │
│             │                                          │                 │
│             │         ┌────────────────────┐           │                 │
│             │         │  RAG pipeline      │◀──────────┘                 │
│             │         │  embed → retrieve  │                             │
│             │         └─────────┬──────────┘                             │
│             │                   │                                        │
│             ▼                   ▼                                        │
│   ┌─────────────────────────────────────────────┐                        │
│   │ ChromaDB (persistent, local)                │                        │
│   │ collection: industrial_documents (cosine)   │                        │
│   │ embeddings: all-MiniLM-L6-v2 (local)        │                        │
│   └─────────────────────────────────────────────┘                        │
└───────────────┬──────────────────────────────────────┬───────────────────┘
                │  http://127.0.0.1:11434              │  local disk
                ▼                                      ▼
   ┌─────────────────────────────┐      backend/data/chroma
   │  Ollama (local inference)   │      backend/data/documents
   │  · qwen3-coder:latest (LLM) │
   │  · llava:latest   (vision)  │
   └─────────────────────────────┘
```

**Request flow:** `User → Frontend → FastAPI → Model Router → RAG / Vision / LLM → Response`

### Components

- **Next.js frontend** — App Router application under `src/`. Each workbench module is a route in
  `src/app/(app)/`. All backend calls go through the typed client in `src/lib/api.ts`.
- **FastAPI backend** — `backend/main.py` mounts one router per service, applies CORS for
  `http://localhost:3000`, and exposes `GET /health`. Interactive API docs are served at `/docs`.
- **ChromaDB vector database** — persistent local store (`backend/data/chroma`), single collection
  `industrial_documents` using cosine similarity. Telemetry disabled.
- **Embeddings** — `sentence-transformers` model `all-MiniLM-L6-v2`, loaded in-process and run
  entirely on local CPU/GPU. No embedding API is called.
- **Ollama** — local model server on port `11434`. The backend calls `/api/generate`, `/api/chat`
  and `/api/tags` over HTTP on loopback only.
- **LLaVA** (`llava:latest`) — vision-language model used for inspection-image analysis; returns
  structured JSON rather than free text.
- **Qwen3-Coder** (`qwen3-coder:latest`) — general reasoning/generation model used for assistant
  answers, RAG answer synthesis and the agent's final assessment.
- **Model routing** — keyword/image-based classifier that picks one of three pipelines and reports
  the decision back to the UI (see [Model Routing](#model-routing)).
- **Agentic workflow** — the maintenance agent chains three steps (vision → retrieval → synthesis),
  passing structured evidence forward at each stage and labelling what was observed, what the
  manual says, and what was inferred.

---

## Technology Stack

| Layer | Technologies |
| --- | --- |
| **Frontend** | Next.js 16.3.4 (App Router), React 19.2.8, TypeScript 5, Tailwind CSS 4, Radix UI primitives, Framer Motion, Lucide icons, ESLint |
| **Backend** | Python 3.12, FastAPI 0.115, Uvicorn 0.34, Pydantic 2, httpx, python-multipart, python-dotenv |
| **AI / ML** | Ollama, `qwen3-coder:latest` (LLM), `llava:latest` (vision), `sentence-transformers` with `all-MiniLM-L6-v2` (embeddings) |
| **Database / vector store** | ChromaDB 1.0 (persistent, local filesystem), cosine distance |
| **Document processing** | PyMuPDF 1.25 (PDF text extraction, page-aware chunking) |
| **Infrastructure** | Local filesystem for uploads, loopback-only HTTP, CORS-restricted origins |
| **Development tools** | npm, `tsc --noEmit`, `next build`, ESLint 9, Git |

---

## How the RAG Pipeline Works

```
PDF upload
   │  POST /documents/upload  (PDF only, ≤ 50 MB)
   ▼
Text extraction          PyMuPDF reads every page; pages with no text are skipped
   ▼
Chunking                 sliding window: CHUNK_SIZE=1000 chars, CHUNK_OVERLAP=200
                         each chunk keeps metadata {doc_id, document_name, page, chunk_idx}
   ▼
Embedding                all-MiniLM-L6-v2 encodes each chunk locally
   ▼
ChromaDB                 chunks + embeddings + metadata written to
                         backend/data/chroma  (collection: industrial_documents)
   ▼
───────────────────────── later, at query time ─────────────────────────
   ▼
Similarity retrieval     POST /rag/query → embed query → top-K (default 5) nearest chunks
   ▼
Prompt assembly          each chunk is wrapped as [Source: <name>, Page <n>] and joined
   ▼
LLM                      Qwen3-Coder receives a system prompt that forbids inventing facts
                         and requires citing document + page (or stating "not found")
   ▼
Grounded answer          response returned with the structured source list
                         { document, page, snippet, score }
```

Key properties:

- **Page-level attribution** — citations point at a specific page, not just a filename.
- **Honest refusal** — if retrieval returns nothing relevant, the model is instructed to say the
  information is not in the indexed documents rather than guess.
- **Scores** — ChromaDB cosine distances are converted to a 0–1 similarity score for display.

---

## Model Routing

`POST /model-router/route` classifies every request and executes the matching pipeline.
The classifier is deterministic (image presence + document keyword matching) so the decision is
explainable and reproducible.

| # | Condition | Request type | Route | Model |
| --- | --- | --- | --- | --- |
| 1 | An image file is attached | `IMAGE` | `VISION` | `llava:latest` |
| 2 | Prompt contains document/maintenance keywords | `DOCUMENT` | `RAG + QWEN` | `qwen3-coder:latest` (over retrieved context) |
| 3 | Otherwise (plain text) | `TEXT` | `DIRECT` | `qwen3-coder:latest` |

Document keywords include: `maintenance`, `manual`, `sop`, `procedure`, `inspection`, `schedule`,
`specification`, `datasheet`, `report`, `certificate`, `guideline`, `standard`, `regulation`,
`compliance`, `safety`, `corrosion`, `calibration`, `vibration`, `lubrication`,
`preventive maintenance`, `corrective maintenance`, `replacement procedure`, `repair procedure`.

Each response reports `request_type`, `selected_route`, `selected_model`, `routing_reason`,
`processing_path` (ordered pipeline steps), `duration_ms` and `local_processing: true`.

`POST /model-router/classify` performs the same classification **without** executing the pipeline —
useful for previewing a routing decision instantly.

---

## AI Agent / Maintenance Assessment

**Endpoint:** `POST /agents/maintenance-assessment`
**Form fields:** `file` (inspection image), `question`, `equipment`

The Pump P-102 scenario combines three evidence sources into one assessment:

```
inspection image  ──►  LLaVA  ──────────────►  Visual Evidence
                                                    │
maintenance manual ──►  chunk → embed → Chroma ──►  Knowledge Evidence
(Pump P-102 PDF)                                    │
                                                    ▼
                                          Qwen3-Coder reasoning
                                                    │
                                                    ▼
                                        Combined Assessment + Next Action
```

1. **Vision step** — LLaVA analyses the photograph and returns
   `{detected_issue, severity, confidence, visual_evidence, recommended_action}`.
2. **Retrieval step** — `"<equipment> <question>"` is embedded and queried against ChromaDB;
   top-K chunks become the Knowledge Evidence block with page citations.
3. **Synthesis step** — Qwen3-Coder receives both evidence blocks under a strict system prompt that
   requires it to separate **(A)** what LLaVA saw, **(B)** what the manual says, **(C)** what it
   infers, to flag conflicts between the two, and to state that qualified personnel must verify
   findings before any action is taken.

The response returns `evidence_sources` (document + page + score) and `models_used`, so the reviewer
can see exactly which models and which manual pages produced the recommendation.

---

## Security and Privacy

- **On-premise processing** — all HTTP calls are to `127.0.0.1`. No cloud AI endpoint is configured
  or contacted.
- **Local Ollama inference** — `qwen3-coder:latest` and `llava:latest` run on the local Ollama server.
  There is no external model API requirement.
- **Local data handling** — uploaded PDFs go to `backend/data/documents/`, vectors go to
  `backend/data/chroma/`. Both live on local disk and both are excluded from Git.
- **Security Center** — `GET /security/status` reports live posture: network mode, Ollama status,
  which models are installed, ChromaDB state, embedding-model availability, CORS origins, document
  and chunk counts.
- **Source citations / auditability** — RAG and agent responses carry document name, page number and
  similarity score, so every claim can be traced back to a page of a local document.

**What is *not* claimed:** there is no authentication or authorisation layer, no encryption at rest,
and no persistent audit trail. The Security Center itself reports
`audit_logging: "LIMITED - Activity logs displayed in UI, no persistent audit trail"`.
The Activity Logs page currently shows sample data.

> **Note on the embedding model:** `all-MiniLM-L6-v2` is downloaded once from Hugging Face the first
> time RAG is used, then served entirely offline from the local model cache. Inference itself never
> leaves the machine. Pre-cache it before going fully air-gapped.

---

## Project Structure

```
sovereign-ai/
├── src/                          # Next.js frontend
│   ├── app/(app)/                # one route per workbench module
│   │   ├── page.tsx              # Overview dashboard
│   │   ├── assistant/page.tsx    # AI Assistant (direct + RAG)
│   │   ├── knowledge/page.tsx    # Knowledge Base / RAG queries
│   │   ├── documents/page.tsx    # upload / list / delete documents
│   │   ├── image-analysis/       # LLaVA vision analysis
│   │   ├── agents/page.tsx       # maintenance assessment agent
│   │   ├── model-router/         # routing inspector
│   │   ├── security/page.tsx     # Security Center
│   │   ├── activity/page.tsx     # Activity Logs (sample data)
│   │   └── settings/page.tsx     # Settings
│   ├── components/               # ui/, layout/, documents/, charts/
│   ├── lib/api.ts                # typed backend client (all endpoints)
│   ├── lib/utils.ts, lib/store/
│   └── data/                     # mock + settings sample data
├── public/                       # static assets
├── backend/
│   ├── main.py                   # FastAPI app, CORS, router mounting, /health
│   ├── config.py                 # pydantic-settings, all tunables
│   ├── requirements.txt          # pinned Python dependencies
│   ├── .env.example              # environment template (committed)
│   ├── create_test_pdf.py        # generates the Pump P-102 test manual
│   ├── services/
│   │   ├── ollama.py             # /ollama  — generate, chat, tags, health
│   │   ├── documents.py          # /documents — upload, extract, chunk, index, delete
│   │   ├── vectorstore.py        # ChromaDB + sentence-transformers wrapper
│   │   ├── rag.py                # /rag/query — retrieve + ground + generate
│   │   ├── vision.py             # /vision/analyze — LLaVA structured output
│   │   ├── maintenance_agent.py  # /agents/maintenance-assessment — 3-step agent
│   │   ├── router.py             # /model-router — classify + route
│   │   └── security.py           # /security/status — live posture
│   └── data/                     # LOCAL ONLY (gitignored)
│       ├── chroma/               # ChromaDB persistence
│       └── documents/            # uploaded PDFs
├── package.json                  # frontend scripts + dependencies
├── package-lock.json
├── tsconfig.json
├── next.config.ts
├── eslint.config.mjs
├── postcss.config.mjs
├── opencode.json
├── .gitignore
├── AGENTS.md / CLAUDE.md
└── README.md
```

---

## Prerequisites

| Requirement | Version / notes |
| --- | --- |
| **Node.js + npm** | Node 20+ (developed on Node 24, npm 11) |
| **Python** | 3.12 (developed on 3.12.10) |
| **Git** | any recent version |
| **Ollama** | Installed and running locally — <https://ollama.com/download> |
| **Ollama models** | `qwen3-coder:latest` (~18.6 GB) and `llava:latest` (~4.7 GB) |
| **Disk / RAM** | ~25 GB free for models; 16 GB RAM recommended for comfortable inference |

Check what you already have:

```bash
node --version
npm --version
python --version
git --version
ollama --version
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/sovereign-ai.git
cd sovereign-ai
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Create the Python environment

**Git Bash / macOS / Linux:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

**Windows PowerShell:**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

> The backend **must** be started from inside `backend/` (its modules import as
> `config` and `services.*`).

### 4. Configure environment variables

```bash
cd backend
cp .env.example .env
cd ..
```

`backend/.env.example`:

```ini
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3-coder:latest
OLLAMA_TIMEOUT=120
CORS_ORIGINS=["http://localhost:3000"]
```

Optional overrides recognised by `backend/config.py` (defaults shown):

```ini
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DIR=data/chroma
DOCUMENTS_DIR=data/documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
VISION_MODEL=llava:latest
MAX_IMAGE_BYTES=20971520
```

The frontend reads `NEXT_PUBLIC_API_URL` from a root-level `.env.local`
(defaults to `http://127.0.0.1:8000` if unset):

```bash
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local
```

> `.env` files are gitignored and are **never** committed.

### 5. Install the Ollama models

```bash
ollama pull qwen3-coder:latest
ollama pull llava:latest
```

Verify:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

> **These model weights are installed locally and are NOT committed to GitHub.**
> The repository contains only the *code* that talks to Ollama — you must pull the models yourself
> after cloning.

---

## Running the Project Locally

You need **three** terminals.

### Terminal 1 — Ollama

Start the Ollama desktop/background service (or `ollama serve`). Confirm it is up:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Terminal 2 — FastAPI backend

```bash
cd backend
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

### Terminal 3 — Next.js frontend

```bash
npm run dev
```

### Open the application

| Service | URL |
| --- | --- |
| **Application** | <http://localhost:3000> |
| Backend API | <http://127.0.0.1:8000> |
| Interactive API docs (Swagger) | <http://127.0.0.1:8000/docs> |
| Backend health check | <http://127.0.0.1:8000/health> |
| Ollama | <http://127.0.0.1:11434> |

Verify the backend:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/security/status
```

---

## Demo Workflow

1. **Open the application** — <http://localhost:3000>. The Overview page shows the workbench modules.
2. **Test the AI Assistant** (`/assistant`) — ask *"What is a centrifugal pump?"*.
   Expect a direct Qwen answer with no citations (plain-text route).
3. **Upload a document** (`/documents`) — upload `backend/create_test_pdf.py`'s output, or any PDF.
   A successful upload reports `Indexed` with a page count and chunk count.
4. **Test RAG** (`/knowledge`) — ask something contained in that PDF, e.g.
   *"What is the rated flow of Pump P-102?"*
   Expect a grounded answer with source cards showing document name, page and score.
5. **Test image analysis** (`/image-analysis`) — upload an inspection photograph and run analysis.
   Expect a structured result: detected issue, severity, confidence, visual evidence, recommended
   action. LLaVA responses can take 30–120 s on CPU.
6. **Test the AI Agent** (`/agents`) — the Pump P-102 assessment:
   - **Equipment:** `Pump P-102`
   - **Image:** an inspection photo of the pump
   - **Question:** *"Assess the condition of the pump and recommend the next maintenance action."*
   Expect a combined assessment with `evidence_sources` (manual pages), `models_used`
   (`llava:latest` + `qwen3-coder:latest`) and a `caveats` field.
7. **Test the Model Router** (`/model-router`) — submit three inputs and watch `routing_reason`
   change: an image, a maintenance question, and a general question.
8. **Show the Security Center** (`/security`) — confirm `ON-PREMISE`, `EXTERNAL AI APIS: NONE
   CONFIGURED`, `DATA PROCESSING: LOCAL`, Ollama `ONLINE`, both models `AVAILABLE`,
   ChromaDB `ONLINE`.

---

## Example Model Routing Demo

| # | Input | Expected route | Expected model | Why |
| --- | --- | --- | --- | --- |
| 1 | Image: `pump_seal_leak.jpg`, question *"What is wrong here?"* | `VISION` | `llava:latest` | Image attached — routed to the vision model first. |
| 2 | *"What is the recommended maintenance interval for Pump P-102?"* | `RAG + QWEN` | `qwen3-coder:latest` | Contains `maintenance` + `interval` → RAG retrieval, then grounded generation. |
| 3 | *"Explain how a centrifugal pump works"* | `DIRECT` | `qwen3-coder:latest` | No document keywords → straight to Qwen with no retrieval. |

**Expected output fields:** `request_type` (`IMAGE` / `DOCUMENT` / `TEXT`), `selected_route`,
`selected_model`, `routing_reason` (e.g. *"Query contains document-related terms (maintenance,
interval) — routed to RAG pipeline"*), `processing_path`, `duration_ms`, `local_processing: true`.

---

## Troubleshooting

**Port 8000 already in use**
```bash
# Windows
netstat -ano | findstr :8000
# macOS/Linux
lsof -i :8000
```
Either stop the process or run on another port and point `NEXT_PUBLIC_API_URL` at it:
```bash
uvicorn main:app --reload --port 8001
```

**Ollama not running** — backend returns `503 Ollama is not reachable.`
```bash
ollama serve          # or launch the Ollama app
curl http://127.0.0.1:11434/api/tags
```

**Model not installed** — Security Center shows `vision_model_status: MISSING`, or Ollama returns 404.
```bash
ollama pull qwen3-coder:latest
ollama pull llava:latest
ollama list
```

**Frontend cannot reach the backend** — check all of these:
- backend is running on port 8000 (`curl http://127.0.0.1:8000/health`)
- root `.env.local` sets `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`
- `CORS_ORIGINS` in `backend/.env` is exactly `["http://localhost:3000"]` (JSON array)
- restart `npm run dev` after changing `.env.local`

**LLaVA inference is slow / times out (`504`)** — normal on CPU; a 4.7 GB model can take minutes.
Raise the timeout in `backend/.env`:
```ini
OLLAMA_TIMEOUT=600
```
Close other GPU/CPU-heavy apps, and confirm the model is already pulled so it is not cold-loading.

**Embedding model fails to load** — `all-MiniLM-L6-v2` downloads from Hugging Face on first use.
Check network access once, or pre-seed the cache, then restart the backend.
Security Center exposes `embedding_model_available` to confirm.

**Environment configuration problems**
- `backend/.env` must live in `backend/`, not the repo root (settings load `env_file=".env"`).
- Run the backend from `backend/` — `main.py` imports `config` and `services.*` as top-level modules.
- `.env` values are re-read with `load_dotenv(override=True)`; restart uvicorn after edits.

**`npm run lint` reports errors** — the repository currently has 4 React-hooks lint errors
(`react-hooks/set-state-in-effect`, `react-hooks/refs`) inherited from the prototype. They do not
block `npm run build` or runtime, but should be cleaned up.

---

## GitHub Repository Notes

The repository is intentionally small (under 1 MB of tracked content). Generated and machine-local
artifacts are excluded via `.gitignore`:

| Excluded | Why |
| --- | --- |
| `node_modules/` | Reinstallable with `npm install` |
| `.next/` | Build output |
| `.venv/`, `venv/`, `__pycache__/`, `*.pyc` | Python environment and bytecode |
| `.env`, `.env.local`, `.env.*.local` | Machine-local configuration — **never committed** |
| `backend/data/chroma/` | Local vector database |
| `backend/data/documents/` | Uploaded documents (potentially sensitive) |
| `*.log`, `.DS_Store` | Logs and OS junk |
| `*.tsbuildinfo` | TypeScript incremental build cache |
| `test_inspection.png` | Scratch output from a test run |
| **Ollama model weights** | Not in the repo at all — pull them locally with `ollama pull` |

Only source code, configuration templates (`backend/.env.example`) and lockfiles are committed.
A fresh clone needs `npm install`, `pip install -r requirements.txt` and `ollama pull` to run.

---

## Current Prototype Status

**Implemented and working against the live backend:**

- AI Assistant — direct Qwen generation and RAG answers with citations.
- Knowledge Base — natural-language RAG queries with source cards.
- Documents — PDF upload, page-aware extraction, chunking, local embedding, ChromaDB indexing,
  listing and deletion.
- Image Analysis — LLaVA structured defect reports with severity/confidence.
- AI Agents — three-step maintenance assessment (vision → retrieval → synthesis) with evidence
  sources, models used and caveats.
- Model Router — live classification plus full pipeline execution with routing reasons and timings.
- Security Center — live posture from the backend, including installed-model checks.
- Backend API — 14 endpoints, Swagger docs at `/docs`, health check at `/health`.

**Partially implemented / sample data (do not present as production behaviour):**

- **Activity Logs** (`/activity`) — displays sample data from `src/data/mock.ts`; there is no
  persistent audit trail.
- **Settings** (`/settings`) — static role/configuration display; changes are not persisted.
- **Overview** (`/`) — some infrastructure cards are illustrative.

**Known limitations:**

- Document **metadata** lives in process memory (`_store` in `services/documents.py`), so the
  document *list* resets when the backend restarts. The ChromaDB chunks themselves persist on disk.
- PDF is the only supported upload format; images are analysed separately, not indexed for RAG.
- No authentication, authorisation, encryption at rest or user accounts.
- No automated tests or CI pipeline yet.
- 4 ESLint errors remain (see Troubleshooting).

---

## Future Improvements

- Persist document metadata in SQLite/Postgres so the list survives restarts.
- Add authentication (OIDC or local accounts) and role-based access control.
- Write a persistent, append-only audit log for every inference request.
- Stream responses (SSE) for lower perceived latency.
- Support DOCX/TXT/CSV ingestion and OCR for scanned PDFs.
- Hybrid retrieval (BM25 + vectors) with a reranking step.
- Docker Compose packaging for one-command deployment.
- Automated test suite and CI (typecheck + lint + backend tests) on push.
- Quantised model presets for lower-spec site hardware.
- Replace sample Activity Logs with real event capture.

---

## Hackathon Context

Sovereign AI was developed as a **completed working prototype for Smart India Hackathon (SIH) 2026**.
It demonstrates a full on-premise agentic AI workflow — vision, retrieval-augmented generation,
multi-step assessment and explainable model routing — running entirely against local Ollama models
with no cloud AI dependency.

---

## License

No license file has been added to this repository yet. One can be added later once the appropriate
licence is chosen.
