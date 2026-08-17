# ARC

Agent Research Copilot (ARC) is a research toolkit for theoretical
physics. It supports paper discovery and analysis, research-domain
construction, proposer-reviewer loops, translation, companion readers, and
source-aware calculation workflows.

ARC is designed for coding-agent hosts such as Codex, Claude Code, and similar
agents.

## Who ARC is for

Use ARC when you need to:

- acquire, parse, search, or summarize research papers;
- construct a source-aware research domain from seed literature;
- run proposer-reviewer idea or calculation loops;
- translate scientific sources or build chapter-aware companion readers; or
- use durable, inspectable research workflows instead of ad hoc
  prompts.

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

### Codex

```bash
codex plugin marketplace add tririver/arc --ref stable
codex plugin add arc@arc
```

### Claude Code

```text
/plugin marketplace add tririver/arc@stable
/plugin install arc
```

### DeepSeek Harness

Install the optional DSH bundle directly from GitHub:

```bash
dsh plugin --profile arc add github:tririver/arc
```

For local development, install a checkout instead:

```bash
dsh plugin --profile arc add /path/to/arc
dsh --profile arc --dump-config
```

The adapter registers the existing ARC Skill and its resource tree. It targets
Linux, macOS, and WSL because ARC's portable runtime launcher is Bash-based.
The bundle has no dependencies and does not make Node.js or a JavaScript
package manager a requirement for ARC users on other agent hosts. With DSH's
default workspace-write sandbox, launch from a writable project directory and
set `ARC_HOME="$PWD/.arc"` so the lazy ARC runtime is installed inside that
workspace.

### Other coding agents

Give your coding agent this repository and ask it to inspect the repository and
install ARC for its environment.

## Start with ARC

After installing ARC, ask for the research outcome directly:

```text
Use ARC to summarize a paper.
Use ARC to build a domain from arXiv:0911.3380 with new papers since 2024.
Use ARC to develop and review ideas from the resulting domain.
Use ARC to check this calculation.
```

An installed ARC plugin exposes its bundled Skill and manuals to the agent.
Those manuals provide task-oriented quick starts; built-in `--help` provides
exact commands, options, and error guidance.

## Development and release

ARC development requires Python 3.11 or newer. Read `AGENTS.md` before making
changes.

Keep research runs and generated output below the git-ignored `local/` tree.

Run focused package tests first, then the combined offline test and build
checks:

```bash
python -m pytest --import-mode=importlib packages/*/tests
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
