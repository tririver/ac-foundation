# ARC

Agent Research Copilot (ARC) is an angentic research toolkit for theoretical physics knowledge domain construction, idea generation and calculation workflows. It works as a plugin of coding agents such as Codex / Claude Code, with the strength of bringing coding agents into a research context, and generating publication-level ideas in theoretical research.

ARC is CLI-first. Its workflow Skill uses the supported package CLIs:

- `arc-paper`: content-addressed paper sources, INSPIRE metadata, unified
  HTML/Markdown/TeX/PDF parsing, reconciliation, and durable paper workflows.
- `arc-domain`: builds a cached research-domain package from a seed paper and
  optional scientific intent.
- `arc-llm`: reusable host LLM execution and provider selection.
- `arc-proposer-reviewer`: typed proposer-worker and reviewer batch
  orchestration with read-only committed-round inspection.
- `arc-translate`: reusable language detection, bilingual glossary generation,
  and source-block translation over verified `arc-paper` documents.
- `arc-companion`: builds source-faithful, chapter-aware PDF and static-web
  original/translation/commentary readers for papers, lecture notes, and books
  from a paired rich source and PDF.
- `arc-jobs`: protocol-neutral inspection, stop requests, and validation for
  durable ARC runs.
- `plugins/arc/skills/arc`: agent-facing workflow instructions for domain
  building, idea generation, and research calculations.

## Who This Is For

Use ARC when you want to:

- Look up reliable paper metadata, references, and citers.
- Parse and reconcile local or provider-acquired paper sources.
- Search text and inline/display mathematics in typed parsed documents.
- Summarize any parsed document through a durable `arc-jobs` workflow.
- Detect source language, build an approximate bilingual glossary, or
  translate verified source blocks independently of Companion rendering.
- Generate Chinese-by-default companion-reading PDF and static-web readers with
  chapter guides, a unified glossary, and an original/translation/commentary
  sequence while retaining source equations, figures, tables, links, and
  bibliography.
- Build a research-domain overview from a seed paper.
- Generate ideas using domain context and reviewer scoring.
- Plan and execute a careful symbolic or numerical research calculation with
  explicit provenance and checks.

Deterministic paper queries do not need an LLM. Paper summaries, domain
briefings, idea loops, and calculation workflow runners need a host LLM
provider.

## Install

### Remarks:

- Permission: the same as many heavy skills/plugins, ARC will need permissions to run Python scripts. Accepting permissions could be annoying. We recommend installing ARC within docker or a virtual machine, and allow all permissions in that virtual environment. As always for working with AI agents, be aware of risk to your data and system. 

- Token usage. As measured using Claude + DeepSeek, a typical run of domain build + idea generation consumes about 1M uncached input tokens, and 0.5M output tokens, in about an hour's running time. The token usage may vary depending on the specific tasks and LLM used. Be aware of token usage and costs. 

- If ARC has played a role in your research, please consider citing the ARC manual.

### Citation

Yanjiao Ma, Yi Wang, and Xingkai Zhang. _ARC: An LLM-Native Agent
Workflow for Theoretical Physics Research_. ChinaXiv:202606.00234, 2026.
https://chinaxiv.org/abs/202606.00234

```bibtex
@misc{ma2026arc,
  title         = {{ARC}: An {LLM}-Native Agent Workflow for Theoretical Physics Research},
  author        = {Ma, Yanjiao and Wang, Yi and Zhang, Xingkai},
  year          = {2026},
  month         = jun,
  publisher     = {ChinaXiv},
  eprint        = {202606.00234},
  archivePrefix = {ChinaXiv},
  url           = {https://chinaxiv.org/abs/202606.00234},
  note          = {Version 1}
}
```

### Requirements:

- Python 3.11 or newer.
- `uv` for fast first-time CLI runtime setup; Python `venv` + `pip` is the
  fallback.
- Network access for first-time INSPIRE/ar5iv fetches.
- Codex, Claude Code, or Kimi Code for supported host LLM work.
- Optional for Markdown report export with Pandoc (see
  `plugins/arc/skills/arc/rules/math_typeset.md`): `pandoc`, `xelatex`, and a
  CJK-capable font such as `Noto Sans CJK SC`.
- For `arc-companion build`: `latexmk`, `xelatex`, Poppler command-line tools,
  and fonts covering the source and annotation languages.

### Agent Plugin Setup

ARC can be installed as a host plugin from this repository. `plugins/arc/` is
the plugin root for both Codex and Claude Code, and
`plugins/arc/skills/arc/` is the single canonical Skill source. The base plugin
contains no MCP manifest or MCP dependency.

Install for Codex (run in shell, or in Codex with `!` prefix):

```bash
codex plugin marketplace add tririver/arc --ref stable
codex plugin add arc@arc
```

Install for Claude Code (run in Claude Code):

```bash
/plugin marketplace add tririver/arc@stable
/plugin install arc
```

The plugin exposes `arc-paper`, `arc-domain`, `arc-llm`, `arc-translate`,
`arc-companion`, `arc-jobs`, and `arc-runtime`. Its isolated core profile also
includes `arc-proposer-reviewer`, which is intentionally invoked through
`arc-runtime arc-proposer-reviewer ...` rather than a plugin-bin wrapper. On
first CLI use, `arc-runtime` installs the immutable ARC release into an
isolated core profile under `~/.codex/arc/runtimes`. It never performs a global
`pip install`.

Prewarm or diagnose the core runtime when needed:

```bash
plugins/arc/bin/arc-runtime setup --profile core
plugins/arc/bin/arc-runtime doctor --profile core
```

If that install fails, later starts fail fast with the saved log path and a
short log tail instead of repeatedly retrying a broken partial install. After
fixing the cause, run `arc-runtime setup --profile core --retry`. Marketplace
installs prefer the host-recorded full commit SHA and otherwise use the bundled
immutable `vX.Y.Z` tag; mutable refs such as `main` and `stable` are rejected.
Source checkouts use local `packages/` automatically. `ARC_INSTALL_REPO_ROOT`
and `ARC_INSTALL_SOURCE=local` select another development checkout.

### Standalone Skill Setup

Hosts without plugin support may install or copy `plugins/arc/skills/arc/` as
an Agent Skill. The Skill carries the same launcher and pinned constraints. If
the ARC commands are not on `PATH`, use:

```bash
<skill-dir>/scripts/arc-runtime arc-paper --help
<skill-dir>/scripts/arc-runtime arc-translate --help
<skill-dir>/scripts/arc-runtime arc-proposer-reviewer --help
<skill-dir>/scripts/arc-runtime setup --profile core
```

There is intentionally no install-time hook: the first real CLI call performs
the audited, isolated setup.

Development benchmarks that must not fall back to an installed or cached ARC
copy can set `ARC_REQUIRE_REPO_ROOT` to the checkout root. Workflow scripts then
prepend that checkout's package sources, verify module origins, and fail before
LLM work if any ARC module or workflow file comes from another installation.
First run `<skill-dir>/scripts/arc-runtime setup --profile core`, then use
`python3 <skill-dir>/scripts/verify-source-runtime.py --repo-root
<checkout> --output <record.json>` to capture module, Git working-tree, and
workflow-file provenance. A verifier launched by system Python re-executes
with the installed core runtime Python before loading checkout sources.

Check the launcher directly from a source checkout:

```bash
plugins/arc/bin/arc-paper --help
plugins/arc/bin/arc-translate --help
plugins/arc/bin/arc-jobs --help
plugins/arc/bin/arc-runtime arc-proposer-reviewer --help
plugins/arc/bin/arc-runtime doctor --profile core
```

Use the source install below only for development or local package testing.

### Release Process

ARC releases use explicit versions in Python package metadata and plugin
manifests. GitHub tags or releases do not update those files automatically.
Run the release helper from a clean checkout on the release branch:

```bash
scripts/release-arc.sh 0.2.0
```

The helper checks that the branch is not behind its upstream, that committed
changes exist since the latest `v*` release tag, and that the target tag does
not already exist. It then pauses for Enter before each mutating step, bumps
ARC package/plugin versions, commits the bump, creates `vX.Y.Z` and
performs push dry-runs, pushes the branch and tag, and
moves `stable` to the release commit.

If you abort after changing version files, after the version bump commit, or
after creating the local release tag, rerun the same command. The helper allows
dirty version-file-only resumes, skips the bump commit when the committed files
already match the requested version, and reuses a local `vX.Y.Z` tag that
already points at `HEAD`.

After the script succeeds, create the human-facing GitHub Release from the
`vX.Y.Z` tag. Marketplace users who should track stable releases should add ARC
with the stable ref:

```bash
codex plugin marketplace add tririver/arc --ref stable
claude plugin marketplace add tririver/arc@stable
```

### Source Install

For development and local testing, create one shared virtual environment and
install every package in editable mode:

```bash
git clone <repo-url> arc
cd arc

python3 -m venv "$HOME/.virtualenvs/arc-dev"
. "$HOME/.virtualenvs/arc-dev/bin/activate"
python -m pip install --upgrade pip

python -m pip install -e packages/arc-jobs[test]
python -m pip install -e packages/arc-llm[test]
python -m pip install -e packages/arc-proposer-reviewer[test]
python -m pip install -e packages/arc-paper[test]
python -m pip install -e packages/arc-domain[test]
python -m pip install -e packages/arc-translate[test]
python -m pip install -e packages/arc-companion[test]
```

Check the installed commands:

```bash
arc-paper --help
arc-domain --help
arc-llm --help
arc-proposer-reviewer --help
arc-translate --help
arc-companion --help
arc-jobs --help
```

Run a deterministic smoke test:

```bash
arc-paper extract-paper-ids "Compare arXiv:0911.3380 and hep-th/0601001."
arc-paper get-title arXiv:0911.3380
```

Export a Markdown report to PDF with the canonical Pandoc command in
`plugins/arc/skills/arc/rules/math_typeset.md`.

Run the translation stages independently when a Companion is not needed:

```bash
arc-translate detect-language note.md \
  --project-dir ./local/translate/note --target-language zh-CN
arc-translate build-glossary note.md \
  --project-dir ./local/translate/note --approx-term-count 50
arc-translate translate-blocks note.md \
  --project-dir ./local/translate/note
```

Each command runs only its named stage and requires verified earlier-stage
artifacts in the same project directory. Approximate term counts deliberately
allow extraction headroom and deduplicated underfill. See
`plugins/arc/skills/arc/manuals/arc-translate.md` for artifact, pause, and
source-format contracts.

Build a source-anchored Companion from Markdown, HTML, flattened TeX, or a
paper identifier. A PDF may be supplied as a validator, while the rich source
remains authoritative for content:

```bash
arc-companion build arXiv:0911.3380 \
  --project-dir ./local/companion/0911.3380 \
  --target-language zh-CN --json
arc-companion status --project-dir ./local/companion/0911.3380 --json
arc-companion resume --project-dir ./local/companion/0911.3380 --json
arc-companion stop --project-dir ./local/companion/0911.3380 --json
arc-companion render --project-dir ./local/companion/0911.3380 \
  --format all --json
arc-companion validate \
  --project-dir ./local/companion/0911.3380 --json
```

`build`, `status`, `resume`, `stop`, `render`, and `validate` are the complete
public Companion command set. Completed child LLM work replays within the same
durable run; `render` and `validate` do not start model calls. See
`plugins/arc/skills/arc/manuals/arc-companion.md` for the current source,
translation, rendering, and pause contracts.

Companion preserves source block order, equations, figures, tables, links, and
bibliography while generating chapter guidance, optional translations, and
selective learning notes. The accepted book is rendered into one immutable
PDF/Web release. Use the owning command's durable controls, or the coding-agent
host's background-command facility when the build should not block the
conversation.

## Configure LLM Providers

ARC uses built-in host providers.

Built-in host providers:

- Codex: `codex-cli`
- Claude Code: `claude-cli`
- Kimi Code: `kimi-code-cli` (experimental; Kimi Code CLI `>=0.28.0`)
- Manual fallback: `manual`

The Kimi provider requires the Node.js/TypeScript `@moonshot-ai/kimi-code`
CLI and an existing login created with `kimi login`. ARC talks to `kimi acp`
over stdin/stdout; it does not use `kimi -p`, add an OpenAI-compatible API
provider, or read and manage Kimi credentials itself.

Check what ARC detects:

```bash
arc-llm doctor --provider auto
arc-llm doctor --provider kimi
```

With `--provider auto`, ARC uses only host-native providers: Codex selects
`codex-cli`, Claude Code selects `claude-cli`, Kimi Code selects
`kimi-code-cli`, and unknown hosts select `manual`. Kimi detection uses
`ARC_AGENT_HOST=kimi-code`, the `@moonshot-ai/kimi-code` package name, or a
reliable `kimi` parent-process signal. An explicit `--provider kimi-code-cli`
works under other hosts. ARC does not read URL-based provider definitions or
Kimi credential values, but the Kimi subprocess inherits the user's Kimi Code
home, authentication, configuration, and persistent sessions. Change the run
model through the run config/CLI: `provider` plus `model_tier`, or exact
`model` with an explicit built-in provider.

Kimi Code retains provider-side persistent sessions. ARC may retain a native
resume handle for its durable run, but does not copy, migrate, or delete Kimi sessions.

`kimi-code-cli` is experimental. Before its first call ARC warns:

> `kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, skills, hooks, plugins, MCP, tool permissions, and persistent sessions; it may access the network, run commands, and modify files.`

ARC denies ACP permission and filesystem reverse requests, but that is not a
sandbox: Kimi automation, hooks, plugins, MCP servers, and local tools may act
outside those reverse requests. Review `arc-llm doctor --provider kimi` before
use. Kimi does not report token usage through this ACP integration, so usage fields remain null.

## Use ARC Through An Agent

Codex and Claude Code can install the CLI-first repository plugin directly with
the marketplace commands in the install section. It loads the ARC Skill and
invokes CLI commands without registering an MCP server.

When using the ARC skill, ask the agent in research terms. Examples:

```text
Use ARC to summarize arXiv:0911.3380.
Use ARC to build a domain for arXiv:0911.3380 focused on quasi-single-field inflation observables.
Use ARC to develop ideas about cosmological collider scalar exchange.
Use ARC to plan and execute the task to be planned.
```

Managed ARC workflows use two automation modes:

- `auto`: continue with safe defaults, while preserving visible warnings.
- `interactive`: pause at workflow-defined major milestones.

The skill defaults to `auto` without presenting a startup menu. Ask explicitly
for manual, step-by-step, staged review, or key-step confirmation to use
`interactive`; asking to discuss first creates a pre-run pause. You can steer a
managed run in either direction while it is active, with the change taking
effect at the next safe boundary. Automatic execution stays within the exact
requested scope: a domain-only request stops after the domain, and an ideas-only
request stops after ranked ideas. Direct ARC tool tasks also default to auto.
Neither mode bypasses authorization, safety, duplicate-charge, scientific, or
error-recovery gates.

## Use ARC From The CLI

The CLI is useful for direct paper checks, scripting, debugging, and working
outside an agent host.

### Paper Metadata And Sources

```bash
arc-paper get-metadata arXiv:0911.3380
arc-paper get-references arXiv:0911.3380 --enrich
arc-paper get-citers arXiv:0911.3380 --limit 1000 --sort mostrecent
arc-paper get-citer-count arXiv:0911.3380
arc-paper fetch-arxiv-auto arXiv:0911.3380
arc-paper fetch-arxiv-pdf arXiv:0911.3380
arc-paper import-source note.md
arc-paper parse-local note.md --validator paper.pdf
```

Every invocation emits one `arc.command_result.v2` JSON document. arXiv auto
fetches ar5iv HTML only; PDF acquisition is explicit. Local TeX input is one
already-flattened file.

The path-only CLI does not create model tasks. For the default Markdown+PDF
full-page visual contract, use the public
`arc_paper.MarkdownPDFVisualParseRunner`; it supplies the `arc-jobs`
`RunContext`, full-page renderer, and shared `arc-llm` task service. Terminal
page results—including `unreviewed` provider failures, pauses, and invalid
outputs—are immutable run artifacts and replay without another provider call.

Paper IDs can be written as new arXiv IDs, old arXiv IDs, INSPIRE record IDs,
or DOI IDs:

```text
0911.3380
arXiv:0911.3380
hep-th/0601001
inspire:837197
doi:10.1088/1475-7516/2010/04/027
```

### Paper Workflows

`PaperSummaryService`, `SummaryBatchRunner`, `ReferenceInferenceService`, and
`ReferenceInferenceRunner` are typed Python workflows. They run child LLM tasks
inside the same parent `RunContext`; group unit results are the only batch item
terminal state. Generic status, stop control, and validation come from
`arc-jobs`.

### Research Domains

A domain is a durable, cache-backed generation built from a seed paper plus
optional intent. It contains foundation selection, selected papers, citation
graph data, an HTML network, a paper pack, an evidence pack, and an optional
field briefing. Domain citation data is fixed to INSPIRE; the CLI does not
offer a citation provider, query, sort/filter, or refresh option.

```bash
arc-domain build arXiv:0911.3380 \
  --intent "quasi-single-field inflation observables" \
  --llm-provider auto \
  --model-tier medium

arc-domain status --domain-id <domain-id>

arc-domain get-summary --domain-id <domain-id>

arc-domain get-graph --domain-id <domain-id>
```

`build` reports the durable run ID and domain ID. Use `resume <run-id>`,
`stop <run-id>`, and `validate <run-id>` for run control. A stop pauses the
current attempt; resume continues the same durable run. The catalog's
`latest` run backs `status --domain-id`; `get-*` reads the published `active`
generation. Different trimmed intent strings produce different domain IDs.

### Direct LLM Checks

Most users should call `arc-paper` or `arc-domain` instead of
calling `arc-llm` directly. For a bounded structured task, create a complete
`arc.llm.request.v2` document and use the durable CLI:

```bash
arc-llm generate \
  --request <request.json> \
  --run-root <project-dir>/context/arc-llm \
  --run-id <stable-run-id>
arc-llm status \
  --run-root <project-dir>/context/arc-llm \
  --run-id <stable-run-id>
```

Use `arc-llm resume` only with the returned resume input, and `arc-llm stop`
only for that durable run. A stop pauses the current attempt and resumes the
same durable run. `arc-llm doctor --provider auto` is the supported
provider diagnostic.

### Proposer-Reviewer Batches

`arc-proposer-reviewer` owns typed multi-worker batches; it is a core tool, not
an `arc-llm` subcommand. In a plugin or standalone Skill installation, invoke
it through the runtime launcher:

```bash
<skill-dir>/scripts/arc-runtime arc-proposer-reviewer inspect \
  --run-root <run-root> --run-id <run-id>
<skill-dir>/scripts/arc-runtime arc-proposer-reviewer trace \
  --run-root <run-root> --run-id <run-id>
<skill-dir>/scripts/arc-runtime arc-proposer-reviewer show-round \
  --run-root <run-root> --run-id <run-id> --loop-id <loop-id> --round <number>
```

These are read-only, no-model-call queries. `inspect` is available for every
run lifecycle and reports best-effort activity that must not guide ranking,
recovery, retries, or resume. `trace` returns only verified, atomically
committed round references plus the run and per-loop revision vector; it never
claims a globally linearized snapshot. A partial published round remains
invisible until its loop commits it. `show-round` is the only public query that
returns committed proposal and review JSON. The public projection does not
expose sessions, task IDs, group IDs, pause records, or physical paths; trace
and round expansion fail closed when verification fails.

## Background Jobs

Long-running package commands own their durable runs and should be started
directly. Use each owning CLI's status/resume/stop/validate commands when
available. `arc-jobs` is the protocol-neutral low-level surface for a known
run:

```bash
arc-jobs status --run-root <run-root> --run-id <run-id>
arc-jobs stop --run-root <run-root> --run-id <run-id> [--reason TEXT]
arc-jobs validate --run-root <run-root> --run-id <run-id>
```

`arc-jobs` does not submit, list, watch, or execute commands. Use the coding
agent host's background-command facility when a direct package command should
run without blocking the conversation.

## End-To-End Research Workflows

The `plugins/arc/skills/arc` layer turns the package commands into
user-facing research workflows. It writes a project directory with
`context.json` and durable artifacts so results can be inspected and resumed.

Generated workflow project directories are a direct child of the directory where
the agent command was launched: `<launch-cwd>/<safe-dir-name>/context.json`.
They are not under host-internal directories such as `.claude/projects` and are
not wrapped in `arc-output/`.

### 1. Build Domain References

Input: a seed paper and optional intent.

Output includes:

```text
<project-dir>/context.json
<project-dir>/domain/<seed-safe>_domain.html
<project-dir>/domain/<seed-safe>_domain_summary.json
<project-dir>/domain/<seed-safe>_domain_summary.md
<project-dir>/domain/foundation_<foundation-safe>.md
<project-dir>/domain/domain-manifest.json
```

Use this when you need a reliable overview of a local research area before
asking for ideas or calculations.

When several domain packages are exported, the manifest helper makes one typed,
deterministic pair-classification LLM request and records the resulting field
groups in `domain/field-grouping.json`. Malformed or inconsistent grouping
content conservatively merges packages with a warning; a typed LLM pause,
failure, or stop pauses the ideas handoff rather than silently merging.

### 2. Ideas

Input: a not-yet-explicit research request plus built domain context.

The release idea workflow feeds ARC-built domain Markdown to proposers. It
materializes one public `BatchRequest` in the project-local run repository,
then ranks only verified committed proposer-reviewer rounds. The ranker chooses
the best qualified committed round for every succeeded loop, not necessarily
the final round; failed and incomplete loops remain visible in
inspection but are not candidates. Use the durable root and ID rather than
reading internal loop, transcript, session, or artifact paths:

```bash
python3 <skill-dir>/scripts/rank-ideas.py \
  --run-root <project-dir>/ideas \
  --run-id <run-id> \
  --format markdown
```

The workflow writes a ranked task-to-be-planned candidate report:

```text
<project-dir>/ideas/<run-id>/
<project-dir>/ideas/<run-id>/ranked-ideas.md
<project-dir>/ranked-ideas.md
```

The report starts with a compact marked summary for each candidate, then
appends one detail section per idea with all round-by-round referee marks and
selected handoff text: title, idea summary, and calculation plan. It should not
invent novelty claims or hide failed idea history. When evidence is enabled,
the entire batch shares a 24-request ARC-paper interaction budget across
`get-metadata`, `get-references`, `get-citers`, `search-metadata`,
`get-arxiv-table-of-contents`, `get-arxiv-section`,
`search-arxiv-full-text`, and `search-arxiv-equations`. Workers have at most
two automatic interaction rounds; a third request pauses for durable resume.
No worker can recurse through shell, ARC CLI, MCP, or nested LLM calls.

The no-info variant is disabled by default and kept as an opt-in test fixture
for workflow development.

### 3. Plan And Execute A Calculation

Input: one task to be planned, such as an explicit calculation idea or a
source-extracted request.

The calculation workflow starts with two phases, then may loop back from
`calculate` to `plan` when a deferred macro block or blocked step needs
expansion:

1. `plan`: gather evidence, write or update `work-note.md`, promote accepted
   premises, define ready-step boundaries, and maintain rough later steps.
2. `calculate`: record current-step result/status, write planning requests
   when plan or foundation material must change, and execute current detailed
   steps through the calculate workflow runner and proposer-reviewer loops.

Primary outputs:

```text
<project-dir>/work-note.md
<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md
<project-dir>/calculate/<run-id>/work-notes/work-note-v002.md
<project-dir>/calculate/<run-id>/execute/calculate.config.json
<project-dir>/calculate/<run-id>/execute/<calculate-run-id>/state.json
```

`work-note.md` is the human and agent source of truth. It contains notation,
axioms, accepted derived results, ready detailed steps, rough later steps,
calculation status, open questions, revision history, journal, and source audit
trail. Main text explains physics and equation logic; journal records execution
events and human resolutions. Runtime JSON is generated only to drive CLI
execution.

The workflow is deliberately conservative: it requires source evidence,
explicit quantity contracts, independent agreement checks, and recorded
validation history before accepting results.

Each ready calculation step is a deterministic public proposer-reviewer batch.
Each initial attempt or bounded recalculation is an independent one-round
`BatchRequest`; the runner consumes proposal and review content only from its
verified committed round. Calculation status records public batch and loop
identities rather than private worker/session or artifact paths. Blind reference
claims are reviewer-only, and a `two_agree` result locks the two accepted
outputs while recalculating only the remaining proposer. Unresolved or
planning-changing results become a human question or a `plan.md` handoff.

## Caches And Refreshing

ARC is cache-first. Repeated calls usually read local JSON/HTML artifacts
instead of refetching data or rerunning LLM work.

ARC paper uses the user cache directory unless `ARC_PAPER_CACHE` is set:

```text
~/.cache/arc/arc-paper/
~/.cache/arc/arc-domain/
```

Set these environment variables to override cache locations:

```bash
export ARC_PAPER_CACHE=/path/to/arc-paper-cache
export ARC_DOMAIN_CACHE=/path/to/arc-domain-cache
```

Use `--refresh` only for paper commands when you intentionally want fresh
source data:

```bash
arc-paper get-metadata arXiv:0911.3380 --refresh
```

Domain builds are cache-first and do not accept `--refresh`; create a new
durable run only through `arc-domain build`.

Validate durable run state:

```bash
arc-jobs validate --run-root <run-root> --run-id <run-id>
arc-runtime doctor --profile core
```

Useful environment variables:

```text
ARC_AGENT_HOST                    Force host detection, for example codex, claude-code, or kimi-code.
ARC_LLM_IDLE_TIMEOUT_SECONDS      General no-substantive-output timeout (default 1800 seconds).
ARC_CODEX_IDLE_TIMEOUT_SECONDS    Codex idle timeout; overrides the general idle timeout.
ARC_CLAUDE_IDLE_TIMEOUT_SECONDS   Claude idle timeout; overrides the general idle timeout.
ARC_KIMI_BIN                      Kimi Code CLI executable (default kimi).
ARC_KIMI_WORK_DIR                 Working directory for new Kimi ACP sessions (default current directory).
ARC_KIMI_IDLE_TIMEOUT_SECONDS     Kimi idle timeout; overrides the general idle timeout.
ARC_LLM_KIMI_LOW_MODEL            Kimi model alias for the low tier.
ARC_LLM_KIMI_MEDIUM_MODEL         Kimi model alias for the medium tier.
ARC_LLM_KIMI_HIGH_MODEL           Kimi model alias for the high tier.
ARC_PAPER_CACHE                   Override the arc-paper cache root.
ARC_DOMAIN_CACHE                  Override the arc-domain cache root.
ARC_RUNTIME_HOME                  Override private ARC runtime storage (default ~/.codex/arc/runtimes).
ARC_INSTALL_REF                   Override with a full commit SHA or immutable vX.Y.Z tag.
ARC_INSTALL_REPO_ROOT             Select a local development checkout.
ARC_INSTALL_SOURCE                Select auto, local, or git package installation.
XDG_CACHE_HOME                    Base cache directory when ARC-specific cache vars are unset.
```

## Troubleshooting

If a paper query fails:

```bash
arc-paper extract-paper-ids "<your input>"
arc-paper get-metadata <paper-id> --refresh
```

If LLM generation is unavailable:

```bash
arc-llm doctor --provider auto
```

If a domain summary or graph is missing:

```bash
arc-domain status --domain-id <domain-id>
arc-domain build <seed-paper> --intent "<same-intent>"
```

Network integration tests are opt-in because they call external services:

```bash
ARC_RUN_NET_TESTS=1 python -m pytest tests/integration -q
```

True LLM integration tests are also opt-in:

```bash
ARC_RUN_LLM_TESTS=1 ARC_RUN_NET_TESTS=1 \
  python -m pytest \
  packages/arc-llm/tests/test_cli_smoke_integration.py \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

## Developer Notes

This repository is organized as Python packages plus thin agent adapters.

Package boundaries:

- `packages/arc-llm` owns reusable host LLM execution: host detection,
  provider selection, model defaults, direct prompt calls, and durable task
  recovery.
- `packages/arc-proposer-reviewer` owns typed proposer-reviewer orchestration,
  worker/reviewer rounds, dialogue artifacts, and consensus result contracts.
  It uses `arc-jobs` for durable execution and `arc-llm` for model calls.
- `packages/arc-paper` owns deterministic paper data access, ID normalization,
  cache layout, ar5iv parsing, INSPIRE access, paper-summary contracts,
  paper-summary orchestration, full-text search, and summary batches.
- `packages/arc-domain` owns research-domain construction from seed papers:
  foundation selection, domain paper selection, graph artifacts, evidence
  packs, HTML rendering, and domain summaries. It calls `arc-paper` for
  single-paper work and `arc-llm` for LLM work.
- `packages/arc-translate` owns reusable language detection, bilingual
  glossary construction, source-block translation, translation review, and
  their durable artifacts. It calls `arc-paper` for parsing and approximate
  keyword inventory, `arc-jobs` for execution, and `arc-llm` for model work.
- `packages/arc-companion` owns paired-source/PDF chapter orchestration,
  chapter guide generation, translation/guide joining, supervised resume,
  deterministic LaTeX/PDF and static-web rendering, and validation. It
  consumes document and asset caches from `arc-paper` and reusable translation
  stages from `arc-translate`.
- `packages/arc-jobs` owns protocol-neutral persistent CLI execution, status,
  stop control, output capture, and ETA. It has no core package dependency.
- `plugins/arc/skills/arc`, prompts, schemas, and plugin manifests describe or
  wrap package behavior; they should not reimplement package internals.

Development rules:

- Keep ARC general-purpose across theoretical-physics domains. Do not hard-code
  seed papers, author names, subfield labels, or field-specific keyword lists.
- Apply the instruction review gate before changing ARC instructions,
  workflows, prompts, schemas, tests, package behavior, packaging
  metadata, or durable documentation. Changes should be portable across
  supported hosts and compatible with ARC's general-purpose research goals.
- Keep agent instructions portable across Codex, Claude Code, Cursor, GitHub
  Copilot, and similar hosts. Use generic terms such as agent, host, skill
  directory, MCP server, and workflow unless a file is host-specific.
- Keep skills concise. Put detailed workflows and troubleshooting in reference
  files.
- Unit tests must not require network access. Use `ARC_RUN_NET_TESTS=1` only
  for explicit network integration runs.
- Durable docs, skills, prompts, schemas, comments, package metadata, and
  workflow files should be written in English unless there is a specific reason
  to do otherwise.

Focused test command:

```bash
python -m pytest \
  packages/arc-jobs/tests \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-translate/tests \
  packages/arc-companion/tests
```

Full local suite used by this checkout:

```bash
python -m pytest \
  packages/arc-jobs/tests \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-translate/tests \
  packages/arc-companion/tests \
  tests -q
```

When changing packaged skills or workflows, edit
`plugins/arc/skills/arc` only. Codex and Claude load the same plugin skill tree;
there are no packaged skill copies to synchronize.

Useful docs/packaging check:

```bash
python -m pytest tests/test_arc_research_workflow_docs.py -q
```
