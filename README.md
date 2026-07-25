# ARC

Agent Research Copilot (ARC) is a research toolkit for theoretical
physics. It combines deterministic paper access with reusable LLM execution,
research-domain construction, proposer-reviewer loops, translation, companion
readers, and source-aware calculation workflows.

ARC is designed for coding-agent hosts such as Codex, Claude Code, and similar
agents, while each Python package and CLI can also be used independently. The
agent-facing Skill lives in `plugins/arc/skills/arc/`; reusable implementations
live in `packages/`.

## Who ARC is for

Use ARC when you need to:

- acquire, parse, search, or summarize research papers;
- construct a source-aware research domain from seed literature;
- run typed proposer-reviewer idea or calculation loops;
- translate scientific sources or build chapter-aware companion readers; or
- give an agent durable, inspectable research workflows instead of ad hoc
  prompts.

Deterministic paper operations do not require a model. Summaries, domain
briefings, proposal/review loops, translation, and companion generation use a
supported host LLM and may consume substantial tokens.

## Citation

If ARC has played a role in your research, please consider citing the ARC manual.

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

## Install

### Remarks:

- Permission: the same as many heavy skills/plugins, ARC will need permissions to run Python scripts. Accepting permissions could be annoying. We recommend installing ARC within docker or a virtual machine, and allow all permissions in that virtual environment. As always for working with AI agents, be aware of risk to your data and system.

- Token usage. As measured using Claude + DeepSeek, a typical run of domain build + idea generation consumes about 1M uncached input tokens, and 0.5M output tokens, in about an hour's running time. The token usage may vary depending on the specific tasks and LLM used. Be aware of token usage and costs.

ARC requires Python 3.11 or newer. Plugin and standalone-Skill installations
use `uv` when available and fall back to Python `venv` plus `pip`. Network
access is needed for first-time package installation and remote paper fetches.

ARC tools can run Python and supported host-model commands. For an agent
installation, use an isolated development environment and review the host's
permissions before starting model-backed work.

### Agent plugin

Install from the stable marketplace reference.

Codex:

```bash
codex plugin marketplace add tririver/arc --ref stable
codex plugin add arc@arc
```

Claude Code:

```text
/plugin marketplace add tririver/arc@stable
/plugin install arc
```

The plugin lazily installs an isolated ARC runtime on first use. Prewarm or
diagnose it with:

```bash
plugins/arc/bin/arc-runtime setup --profile core
plugins/arc/bin/arc-runtime doctor --profile core
```

### Standalone Skill

Hosts without plugin support may install or copy
`plugins/arc/skills/arc/`. If ARC is not already on `PATH`, run package
commands through the bundled launcher:

```bash
<skill-dir>/scripts/arc-runtime arc-paper --help
<skill-dir>/scripts/arc-runtime arc-proposer-reviewer --help
```

## Start with an agent or CLI

With the ARC Skill installed, ask for the research outcome directly:

```text
Use ARC to summarize a paper.
Use ARC to build a domain from arXiv:0911.3380 with new papers since 2024.
Use ARC to develop and review ideas from the resulting domain.
Use ARC to check this calculation.
```

The Skill selects package commands, manages project-local artifacts, and
explains pauses or recovery. Its quick-start manuals are self-contained; a
package README is optional background, not a runtime dependency.

For direct use, start with the owning CLI's help. Non-help commands return one
typed JSON result on stdout.

| Package | Primary responsibility | Entry point |
| --- | --- | --- |
| [`arc-paper`](packages/arc-paper/README.md) | Paper data, sources, parsing, search, and paper workflows | `arc-paper --help` |
| [`arc-domain`](packages/arc-domain/README.md) | Research-domain construction and exports | `arc-domain --help` |
| [`arc-llm`](packages/arc-llm/README.md) | Host LLM execution and recovery | `arc-llm --help` |
| [`arc-proposer-reviewer`](packages/arc-proposer-reviewer/README.md) | Typed proposer-reviewer batches | `arc-proposer-reviewer --help` |
| [`arc-translate`](packages/arc-translate/README.md) | Scientific language, glossary, and translation stages | `arc-translate --help` |
| [`arc-companion`](packages/arc-companion/README.md) | Source-anchored companion builds and releases | `arc-companion --help` |
| [`arc-jobs`](packages/arc-jobs/README.md) | Durable-run state, inspection, and stop control | `arc-jobs --help` |

Exact commands, options, and error guidance live in `--help`. The Skill manuals
explain when to combine packages into a research workflow.

## Development and release

### Source checkout

For development, create a virtual environment and install the packages in
dependency order:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e packages/arc-jobs[test]
python -m pip install -e packages/arc-llm[test]
python -m pip install -e packages/arc-proposer-reviewer[test]
python -m pip install -e packages/arc-paper[test]
python -m pip install -e packages/arc-domain[test]
python -m pip install -e packages/arc-translate[test]
python -m pip install -e packages/arc-companion[test]
```

Keep research runs and generated output below the git-ignored `local/` tree.

Run focused package tests first, then the combined offline suite:

```bash
python -m pytest packages/arc-paper/tests
python -m pytest \
  packages/arc-jobs/tests \
  packages/arc-llm/tests \
  packages/arc-proposer-reviewer/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-translate/tests \
  packages/arc-companion/tests
scripts/check-packages.sh
```

Network and live-model tests are opt-in. Do not use them as the default
development check.

Releases are explicit human operations from a clean release checkout:

```bash
scripts/release-arc.sh <version>
```

The helper validates the release, updates package and plugin versions, and
pauses before its mutating Git steps. See `AGENTS.md` for repository
development, verification, and release constraints.
