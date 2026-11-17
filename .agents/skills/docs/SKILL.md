---
name: docs
description: Use whenever creating or editing PyMC-Marketing documentation, including small updates to existing notebooks. Apply the repository's documentation conventions and notebook visibility policy.
---

# Docs

When delegating documentation work, include this skill's path in the worker's task.

## Notebook visibility

For every new or edited documentation notebook, review visibility throughout the notebook:

- Show code central to the teaching arc, such as model definitions, key API calls, and essential analysis.
- Use `hide-input` for incidental plotting, formatting, and data manipulation, unless those operations are being taught.
- Use `hide-output` for trivial output, including MCMC messages and progress displays, unless sampler behavior is part of the argument.
- Keep explanatory figures, comparisons, relevant diagnostics, and consequential warnings or unexpected results visible.
- Choose input and output visibility independently; split mixed-purpose cells when necessary.

Add tags to `cell.metadata.tags`, preserving existing tags, code, and stored outputs.
Check the rendered page when practical. For visibility-only edits, validate notebook structure and confirm code and results are unchanged; resampling is unnecessary.
