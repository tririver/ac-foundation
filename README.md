# AC Foundation

AC Foundation provides neutral, reusable infrastructure for agentic products.
It does not define a research or learning workflow and ships no agent-host
Plugin or Skill.

Packages:

- `ac-jobs`: durable runs, artifacts, recovery, work groups, and cooperative stop.
- `ac-llm`: provider-neutral model execution and structured output.
- `ac-document`: document ingestion, parsing, search, and rich-document contracts.
- `ac-proposer-reviewer`: typed proposal and review orchestration.

All packages require Python 3.11 or newer. Install distributions directly or
let a product launcher create a private runtime. Public CLIs are `ac-jobs`,
`ac-llm`, `ac-document`, and `ac-proposer-reviewer`.

## Development

```bash
python -m pytest --import-mode=importlib packages/*/tests tests
scripts/build-packages.sh
```

Generated files, test runs, and caches belong under ignored `local/` paths.
See `AGENTS.md` for repository rules.

## Release

All Foundation distributions share one repository version. Prepare a release
with:

```bash
scripts/release-ac-foundation.sh VERSION
```

Product repositories depend on Foundation with `>=2,<3`; runtime source locks
pin an exact full Git commit SHA.

## License

MIT. See `LICENSE`.
