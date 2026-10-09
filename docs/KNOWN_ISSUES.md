# KNOWN ISSUES

Honest disclosure of defects in this fork, measured at the `c786f9d`-era tree.
This document exists because the previous README claimed "Production Ready" and
"Security 95% Complete" while the code did not support it. These are known, and
most have fixes in progress.

Severity ordering matches this document's own ordering. The full ranked audit,
including the upstream-side findings, lives in the project's audit history.

## 1. GPU inference is mocked, not real — CRITICAL for anyone benchmarking

`src/ct/gpu/boltz2_service.py`:

- `is_available()` returns `True` **unconditionally** — it does not check whether
  any model is installed, loadable, or licensed.
- `load_model()` sets `self._loaded = True` without loading anything.
- Inference returns the literal string `"MOCK_PDB"`.

Consequence: the `503` guard in `src/ct/api/main.py` never fires, so
`/api/v1/gpu/predict-structure`, `/api/v1/gpu/predict-affinity`, and
`/api/v1/gpu/dock` **accept requests and return fake structures**. Any binding,
docking, or affinity number produced by this fork today is a placeholder, not a
result. There is no silent-corruption risk because no real inference runs — but
the endpoints look functional.

**Status:** open. Must be wired to a real service, and `is_available()` must
actually probe for it, before any structural claim is made from this software.

## 2. The `security/` module is largely inert — CRITICAL

Present but not enforced anywhere:

| Component | Location | Reality |
|---|---|---|
| `require_role(role)` | `src/ct/security/auth.py:371` | **Zero call sites.** Defined and exported, applied to nothing. |
| `RateLimiter` | `src/ct/utils/error_handling.py:306` | **Zero call sites.** Only its own docstring example references it. |
| API-key `rate_limit` | `src/ct/security/api_keys.py:26` | Field is persisted (`:99`) and **never read** by `verify_key()`. Worse, the value is client-supplied via `POST /api/v1/api-keys`, so a caller sets their own ceiling. |
| Authentication | `src/ct/api/main.py` | Applied to **5 of 22** endpoints. |

The unauthenticated set includes everything that costs money or CPU:

- `POST /api/v1/query` (`:164`) — single-agent LLM execution
- `POST /api/v1/multi-agent` (`:218`) — fan-out across specialist agents
- `POST /api/v1/dmta` (`:256`) — `iterations ≤ 5` × `num_candidates ≤ 100`, no cost ceiling
- `POST /api/v1/rlef/train` (`:437`) — spawns training via `BackgroundTasks`, no auth, no dedup, no concurrency cap
- `POST /api/v1/gpu/*` (`:617`, `:636`, `:655`)
- `GET /metrics` (`:598`), `GET /api/v1/health/detailed` (`:606`)

Combined with `docker-compose.yml` publishing `"8000:8000"`, any reachable
deployment hands an anonymous caller unmetered vendor-API spend.

**Also:** users, API keys, and the token revocation list are plain in-memory
dicts (`auth.py:71-72`, `api_keys.py:55-56`). Every restart wipes every user and
key; there is no admin bootstrap; multi-worker deployments each hold a different
user table, so a token minted by one worker fails on another.

**Status:** open. Auth must be a `Depends` on every non-public route with a
default-deny posture, and a real limiter must consume the `rate_limit` field
that already exists.

## 3. Hand-rolled JWT is not a MAC — HIGH

`src/ct/security/auth.py:293` and `:324`:

    signature = hashlib.sha256((message + secret).encode()).hexdigest()

No HMAC, no `alg` enforcement on verification, no library. This is
`sha256(header ‖ payload ‖ secret)`, so **length extension applies**: given one
valid signature an attacker can append data and derive a valid signature for the
extended message without knowing the secret. The comparison at `:327` is
`signature_b64 != expected_sig_b64`, a non-constant-time compare.

Compounding: `JWT_SECRET` defaults to `secrets.token_hex(32)` generated **per
process** (`auth.py:21`), and nothing in the repo or compose file ever sets it.
Tokens minted before a restart are invalid after; each worker in a multi-worker
deployment signs with a different key.

**Status:** open. Replace with `pyjwt`, hard-fail startup when `JWT_SECRET` is
unset, compare with `hmac.compare_digest`.

## 4. CORS allows any origin with credentials — HIGH

`src/ct/api/main.py:69-75`:

    allow_origins=["*"]
    allow_credentials=True

Starlette reflects the requesting `Origin` and sends
`Access-Control-Allow-Credentials: true`. Today the API authenticates by
`Authorization` header rather than cookie, which limits the blast radius — but
the moment cookie auth is added (the obvious next step for the existing
`/auth/login` flow), every website an authenticated user visits gains
credentialed cross-origin access. Browsers also treat `null` origin as matching.

**Status:** open.

## 5. `context` is splatted into the agent runner — HIGH

`src/ct/api/main.py:184`:

    result = _run_query(request.query, **(request.context or {}))

`QueryRequest.context` is an unvalidated `dict`. `**` expansion binds any key
matching a named parameter of `run_query`, and `model` is one, so a caller
selects the LLM:

    POST /api/v1/query  {"query":"hi","mode":"single","context":{"model":"<any id>"}}

The attacker picks the model — cheap, old, or attacker-chosen — silently changing
cost and behaviour. Every future named parameter becomes attacker-settable the
same way.

**Status:** open. Validate `context` with a Pydantic model and take `model` from
server config only.

## 6. tar-slip on an unverified download — HIGH

`src/ct/data/drkg_downloader.py:106-108` calls `tar.extractall(self.data_dir)`
with no `filter=` argument and **no integrity check** — no SHA-256, no signature,
no size bound. On Python 3.10/3.11 (`requires-python = ">=3.10"` allows these) a
crafted archive can write outside the target directory. The trust chain is
whoever controls the S3 bucket or its TLS path.

**Status:** open. Pin SHA-256, pass `filter="data"` explicitly, extract to a
fresh temp dir.

## 7. Cypher from LLM output, guarded by a bypassable blocklist — HIGH

`src/ct/knowledge_graph/text_to_cypher.py:199-217` allows queries starting with
`MATCH`/`WITH`/`RETURN`/`CALL` and rejects strings containing `DELETE`/`REMOVE`/`SET`.
Gaps: `CREATE` and `MERGE` are not blocked; `CALL` is explicitly permitted as a
leading clause and `docker-compose.yml:52` loads APOC, so `CALL apoc.load.csv(...)`
is a file-read primitive; comment syntax hides keywords from the substring scan.
When `translation.confidence >= 0.7` the model-authored query is executed directly
(`:265-267`); `EXPLAIN` validation at `:231` does not execute, so a query that
validates still runs unmodified.

**Status:** open. The real control is a **read-only database user**; the
blocklist is not a security boundary.

## 8. Path-traversal in structure tools — MEDIUM

`src/ct/tools/structure_io_tools.py` (`:25`, `:41`, `:71`, `:87`, `:102`) passes
caller-supplied paths straight to `Path(...).read_text()` in
`src/ct/structure_io/pdb_handler.py:70`, `fasta_handler.py:52`, `h5ad_handler.py:50`.
No `resolve()`, no containment check. Upstream does this correctly —
`src/ct/tools/files.py:53-67` resolves symlinks and asserts `relative_to(cwd)`.

**Status:** open. Mirror upstream's `_is_within_cwd`.

## 9. Session traces are world-readable and unbounded — MEDIUM

`src/ct/session_logging/logger.py:100-101`, `:329-334`;
`trace_store.py:80-82`, `:91`; `src/ct/memory/vector_memory.py:73-74`, `:322-323`.
`mkdir` without `mode=0o700` and `open` without explicit permissions → typically
`0755`/`0644`, governed only by umask. Contents are full prompts, responses,
tool arguments, compound SMILES, and target sequences, plus the attacker-supplied
`context` dict from issue 5. `traces` also grows without bound.

Note `src/ct/security/secrets.py:203-208` gets this right (`os.chmod(0o600)`), so
the pattern was known and simply not applied to the higher-volume stores.

**Status:** open.

## 10. Internal exception text returned to clients — MEDIUM

`src/ct/api/main.py:215`, `:253`, `:293`, `:382`, `:397` use
`HTTPException(500, detail=str(e))`. `src/ct/models/llm.py:163-164` returns
`f"Error: {e}"` as response *content*. These leak connection URIs, file paths, and
occasionally credential fragments.

**Status:** open.

## 11. Insecure default credentials across the compose stack — MEDIUM

`docker-compose.yml`: `POSTGRES_PASSWORD=ct` hardcoded, `NEO4J_PASSWORD`
default `changeme`, Grafana default `admin`, `QDRANT_API_KEY` defaults to empty
(auth disabled), `JWT_SECRET` never set. Redis is published `6379:6379` with no
`requirepass`; Neo4j `7474`/`7687`, Postgres `5432`, Qdrant `6333/6334`,
Prometheus `9090`, Grafana `3000` are all published to the host.

Compounding: `src/ct/knowledge_graph/neo4j_client.py` **never reads the
environment** — `Neo4jConfig` hardcodes `password: str = "password"` (`:22`) and
`uri = "bolt://localhost:7687"` (`:21`), so the compose credentials are ignored by
the application entirely.

**Status:** open.

## 12. `tar.extractall` and dependency reproducibility

`requirements.txt` and both `pyproject.toml` files use only `>=` floors with no
lockfile, across a large fast-moving ML stack. `fork/requirements.txt:7` declares
`python>=3.10,<3.13`, which is not an installable requirement — the interpreter
constraint belongs in `requires-python` (which `pyproject.toml:9` does correctly),
so that line is a no-op.

**Status:** open. Commit a lockfile, pin the ML stack, delete the bogus line.

## What is *not* wrong

Recording these so the list above is not read as "nothing works":

- **No live secrets are committed.** Swept both trees for `sk-`, `sk-ant-`,
  `ghp_`, `AKIA`, `xox[baprs]-`, `AIza`, PEM private-key headers: zero hits. Also
  clean across `git log --all -p`. Matches in source are dummies (`"test-key"`,
  `"ct_test123"`).
- The Dockerfile runs as non-root `ctuser` with correct ownership.
- `src/ct/security/secrets.py` handles secret file permissions correctly.