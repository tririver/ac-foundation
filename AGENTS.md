# ARC Development Guidance

ARC is a theoretical-physics research toolkit built from reusable Python
packages and thin agent-facing adapters. Keep package behavior independent of
any particular agent host or checked-out Skill.

## Governing Philosophy: Infrastructure for Model-Led Science

**ARC exists to expand the scientific ability and creativity of capable
models, not to replace their scientific judgment with complicated program
logic.**

- Give models better information, easier access to evidence, reliable tools,
  recoverable execution, and clear provenance.
- Provide simple, repeatedly validated research practices where infrastructure
  is genuinely useful: independent proposer-reviewer criticism, focused
  literature and citation searches, reproducible calculations, explicit
  assumptions and uncertainty, and respectful citation of prior work.
- Treat citation as scientific context and scholarly courtesy. Help models find
  and acknowledge predecessors; do not turn a bounded search result into proof
  of novelty or a qualification gate.
- Leave scientific questions to model reasoning wherever correction, revision,
  comparison, or further evidence can resolve them. Scientific merit,
  relevance, novelty, simplicity, research direction, decomposition, and
  presentation are not validity conditions for program code.
- Use prompts, evidence, warnings, confidence, alternatives, and reviewer
  feedback to improve scientific decisions. Do not encode debatable scientific
  taste as hard routing, deletion, disqualification, closed taxonomies, or
  mandatory methodological bureaucracy.
- Reserve hard program stops for genuine system boundaries: missing authority,
  corrupt or unreadable durable state, invalid machine contracts that remain
  unusable after bounded recovery, unsafe destructive actions, or the complete
  absence of a usable result. Delivery failures must not erase valid scientific
  work.
- Before adding complex enforcement code, ask whether it protects a necessary
  system invariant or instead constrains a scientific choice that a capable
  model should make. If the latter, provide information and feedback and trust
  the model.

## Navigation

- Read `plugins/arc/skills/arc/workflows/` for agent workflows; use the
  documented CLI or public API when the Skill is unavailable.
- Read `packages/` for implementation and package-local `tests/` for focused
  tests. Use repository `tests/` for cross-package contracts.
- Inside this checkout, put every non-source task artifact in an ignored
  `local/` path: development runs, temporary extraction or conversion files,
  render previews, logs, command results, generated artifacts, and evaluation
  output. Do not create top-level `tmp/`, `output/`, `results/`, or similar
  task-data directories. Generic tool or Skill output conventions do not
  override this repository rule. The only exceptions are a project directory
  explicitly supplied by the user and the documented ARC shared runtime and
  paper-cache roots.
- Treat a project directory explicitly supplied by the user as the project
  root itself. Do not append an extra workflow directory or replace it with
  names such as `build-v2`, `fresh`, or attempt-specific variants. When choosing
  a project root inside this checkout, use a stable descriptive path below
  ignored `local/`; `local/` is a checkout convention, not a requirement for
  external user-supplied directories.

Workflows:

- `domain.md`: research-domain build.
- `ideas.md`: idea generation and proposer-reviewer evaluation.
- `plan.md`: evidence-based calculation planning.
- `calculate.md`: premise checks and calculations.
- `check.md`: research-note claim checks.
- `companion.md`: translated source-anchored Companion.

Packages:

- `arc-jobs`: durable job execution.
- `arc-llm`: provider and model calls.
- `arc-proposer-reviewer`: proposal and review orchestration.
- `arc-paper`: paper access, caching, and summaries.
- `arc-domain`: domain discovery, typed summaries, package views, and
  evidence-bearing domain artifacts.
- `arc-translate`: bilingual translation and review.
- `arc-companion`: guide assembly and Companion releases.

## Research Principles

- Build general theoretical-physics infrastructure, not special behavior for a
  paper, author, subfield, or generated site. Avoid hidden hard-coded IDs,
  names, labels, and keyword lists.
- Prefer configurable, documented heuristics. Use example papers only as
  regression cases; they must not define production behavior.
- Domain-specific scientific logic is acceptable when exposed through explicit,
  user-selectable configuration, plugins, or separately owned extensions.
- Preserve provenance, assumptions, approximation regimes, and uncertainty.
  Treat model agreement as evidence, not proof; verify scientific claims and
  calculations in proportion to their risk.
- ARC harnesses serve model reasoning with richer context, evidence,
  recoverability, and reliable publication; they do not exist to narrow model
  creativity. Prompts and schemas may constrain identities, evidence, anchors,
  machine-checkable coverage, publication, and recovery, but must not prescribe
  one interpretive, creative, or pedagogical form. Treat enumerated forms and
  dimensions as optional, non-exhaustive possibilities, never as a quota,
  required taxonomy, or default template.
- Treat exploration profiles as optional lenses, not assignments or coverage
  quotas. The exact user intent remains primary, and a model may leave a profile
  when a simpler direct route better addresses that intent.
- In scientific selection and review, separate an idea's nucleus from fixable
  formulation errors. If an error can be repaired without replacing that
  nucleus, preserve the direction and return actionable feedback. Prefer the
  shortest minimally sufficient setup after asking which assumptions, methods,
  observables, and validations can be removed without making the core result
  undefined, infeasible, non-novel, or scientifically inconsequential.
- Do not turn model-correctable scientific weaknesses into hard
  disqualification. Express them through evidence, marks, uncertainty,
  targeted revision feedback, and visible warnings. Reserve hard stops for
  conditions under which reasoning cannot safely or meaningfully continue,
  such as missing authority, unreadable or invalid contracts, corrupt durable
  state, or the absence of any complete usable output. A new hard stop must
  explain why model revision, retry, advisory reporting, or human choice cannot
  repair the condition.
- Author names are publication identity managed by ARC, not an interpretive or
  creative constraint. Automatically parsed author candidates require model
  verification against the frozen source; publish an attribution only at high
  confidence, otherwise retain an explicitly empty authorship result.

## Local Robustness

- ARC is local research software. Handle interrupted writes, accidental
  corruption, malformed provider output, and ordinary cooperating-process
  races.
- Use the existing cooperative stop semantics for run control. Do not add a
  separate cancel concept without a concrete lifecycle requirement that stop
  cannot satisfy.
- Find the root cause of failures. When a run fails because of a general ARC
  contract or implementation defect, first fix ARC and add a regression test;
  do not hide the defect with ad hoc task-script changes or project-local
  workarounds. Use a local correction only when the cause is genuinely
  specific to that project's input or configuration.
- Treat agents, models, users, and the local filesystem as trusted unless the
  user explicitly introduces a hostile boundary. Use simple guardrails against
  accidental deletion, credential disclosure, unsafe provider invocation, and
  partial state; do not add adversarial security machinery without that
  requirement.

## State and Deliverables

- ARC-owned shared storage is limited to runtimes under `~/.arc/runtimes` and
  the shared `arc-paper` cache under `~/.arc/cache/arc-paper`. A cache may be
  shared only with documented cross-project identity, validity/invalidation,
  concurrent-access, and lifecycle rules; otherwise it is project state below
  `<project-dir>/.arc/`.
- Durable runs, LLM sessions and transcripts, child workspaces, domain state,
  diagnostics, temporary files, and unpublished generations are project state,
  not shared caches. Do not add fallback discovery of legacy shared roots.
- Within this checkout, write runs and generated files to ignored `local/` by
  default. Respect an external project directory supplied by the user.
- Hidden state must not be the only final deliverable. Publish narrative
  research reports as HTML or PDF; retain JSON, CSV, graph data, code, and
  other machine-readable deliverables in native formats.

## Workflow and Change Design

- Let workflows support agent reasoning rather than impose mechanical
  micro-steps. Preserve source context, user intent, constraints, and coverage;
  `plan.md` chooses grouping and execution order.
- Treat a change as non-trivial when it materially affects public behavior,
  persisted state, or package or workflow boundaries, or is expected to span
  multiple functional commits.
- Before implementing a non-trivial feature or capability, identify the
  essential complexity inherent in the requested behavior and separate it from
  accidental complexity introduced by architecture, state machines,
  dependencies, durability, operations, or maintenance.
- First redesign to remove or reduce accidental complexity. Prefer simpler
  boundaries and less state, then check that the remaining design preserves
  generality, host portability, package ownership, and every producer and
  consumer of a changed contract.
- A small, time-bounded, reversible spike is allowed within the user's request
  or an approved design when it has acceptance criteria, uses ignored `local/`
  output, and is not kept as a latent production path.
- Pause for user direction before implementation when the solution exceeds the
  user's request or an approved design, requires an irreversible action,
  introduces a material new dependency, migration, durable-state commitment,
  or operational burden, or has essential complexity materially beyond what
  the approval reasonably implied. Explain the essential need, simpler
  alternatives, affected package boundaries, durable-state implications, and
  verification burden.
- After abandoning an approach, remove its obsolete code paths, tests,
  documentation, and generated artifacts unless they support a current,
  documented fallback or independently used capability. Preserve concise
  negative research findings and decision rationale; keep bulky evidence under
  ignored `local/`.
- Do not reject a requested policy change merely because it differs from current
  policy. Evaluate its soundness, portability, coherence, and maintenance cost.
  If it is unsound, explain the specific conflict and leave files unchanged
  unless the user approves a revised direction. Do not partially implement a
  rejected change.

## Planning, Versions, and Git

- Keep implementation plans under ignored `local/implementation-plans/` and
  never stage them. Before a non-trivial change expected to span multiple
  functional commits, create and maintain
  `local/implementation-plans/<task-slug>.md`, verify Git ignores it, and use it
  as durable execution state across compaction, handoffs, and concurrent work.
- Obtain explicit user approval before changing any release or distribution
  version, including `VERSION`, manifest/package versions, dependency ranges,
  and in-code package versions. Internal schema versions may be updated without
  separate approval when required; update their producers, consumers,
  validation, tests, and docs together.
- Treat the worktree as shared: preserve others' changes, inspect status and
  diffs, and stage only intended paths or hunks.
- After a functional change passes its relevant validation, commit it before
  starting the next functional change or handing work back to the user, unless
  the user explicitly asks not to commit. Do not leave validated changes
  uncommitted merely because unrelated worktree changes are present.
- Do not revert, rewrite, discard, or absorb another agent's work to isolate a
  commit; do not wait for a globally clean worktree or repeatedly diff-chase for
  perfect commit purity.
- Do not create, move, or push Git tags. Use the repository-configured author
  identity for agent commits.

## Package and Host Boundaries

- Do not substitute semantic keys, execution fingerprints, operational policy,
  artifact digests, or resume-input digests for one another.
- Package source must not import, inspect, execute, or derive runtime behavior
  from Skill, plugin, or workflow files. Skills, prompts, schemas, and plugin
  adapters may describe or invoke documented package behavior. Each package
  must remain usable from its installed distribution without an agent host or
  ARC checkout.
- Keep packages, skills, prompts, scripts, schemas, and documentation portable
  across coding-agent hosts. Make host-specific behavior optional and provide
  a portable CLI or public API fallback.
- Keep `SKILL.md` concise and task-oriented: explain when and how to use ARC
  tools, not complex control flow or package internals. Put detailed examples
  and troubleshooting in focused references; when a Skill needs steps, label
  explicit phases and steps.

## Testing and Evaluation

- Run focused tests for touched code first. When practical, run the complete
  offline suite:

  ```bash
  packages/arc-paper/.venv/bin/python -m pytest --import-mode=importlib \
    packages/*/tests tests
  ```

- Unit tests must be offline by default. Network integration tests are opt-in
  through `ARC_RUN_NET_TESTS=1`.
- Prefer deterministic fixtures, fake providers, and bounded non-generative
  checks. Run a live semantic smoke without separate approval only when it is
  materially necessary to check a provider, prompt, schema, orchestration, or
  model-output semantic boundary. Limit it to one worker, at most three total
  provider calls (including retries and formatter calls), five minutes, fixed
  inputs, explicit acceptance criteria, and ignored `local/` output.
- Do not expand it into a domain build, substantial Companion build, broad idea
  generation, benchmark, or other token-expensive live workflow; those require
  explicit user authorization. Do not pause an otherwise complete
  implementation merely to request an optional expensive evaluation; finish
  offline verification and commits, then recommend it in the handoff if useful.

## Language

- User-facing discussion may use the user's language.
- Skills, prompts, schemas, code comments, docstrings, package metadata, and
  durable documentation are English unless a deliverable needs another
  language.
