# AC Foundation Development Guidance

AC Foundation owns neutral infrastructure shared by independent agentic
products. Package code must not depend on ARC, ALC, an agent host, a Plugin,
or a checked-out Skill.

## Boundaries

- `ac-jobs` has no internal package dependency.
- `ac-llm` may depend on `ac-jobs`.
- `ac-document` may depend on `ac-jobs` and `ac-llm`.
- `ac-proposer-reviewer` may depend on `ac-jobs` and `ac-llm`.
- Schemas, imports, distributions, CLIs, and owned environment variables use
  `ac.*`, `ac_*`, `ac-*`, and `AC_*` respectively.
- Runtime state defaults to `~/.ac`; project document cache defaults to
  `.ac/cache/ac-document`.
- Durable runtime identity reports only AC-owned paths and checked-in product
  defaults. Provider subprocesses still inherit the live process environment
  so normal credential and provider configuration mechanisms keep working.
- The canonical product runtime bootstrap and DSH LLM bridge live here.
  Product repositories may carry generated, checksum-verified copies.

## Robustness

Handle interrupted writes, malformed provider output, and cooperating-process
races. Preserve durable work across recoverable delivery failures. Reserve hard
stops for invalid machine contracts, corrupt durable state, missing authority,
unsafe destructive actions, or no usable result.

## Development

- Use ignored `local/` paths for non-source artifacts.
- Unit tests are offline by default; network tests require
  `AC_RUN_NET_TESTS=1`.
- Run focused tests first, then the full suite:

  ```bash
  python -m pytest --import-mode=importlib packages/*/tests tests
  ```

- Build all packages with `scripts/build-packages.sh`.
- Do not change a release version without explicit user approval.
- Preserve unrelated worktree changes; commit each validated functional unit.
