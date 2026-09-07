---
name: work-in-repo
description: Learn how to work in the current folder (repository)
---

# Introduction

The current folder is the github repository of pymc-labs for pymc-marketing.

# How to do when user ask a request?

If the user ask you to build something then:
1. Always run `uv sync` to set up the environment.
2. If gh is missing install using `brew install gh` (macOS) or follow https://github.com/cli/cli#installation
3. Use `uv run` for all Python commands within the project.
4. Follow linter rules and contribution.md recommendations.
