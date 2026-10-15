---
name: docs-notebook-visibility
description: Apply PyMC-Marketing's code and output visibility policy whenever creating or editing a documentation notebook, including small edits to existing notebooks. Review visibility across the edited notebook using hide-input and hide-output cell tags.
---

# Documentation notebook visibility

When creating a notebook in `docs/` or editing an existing one, review cell visibility throughout that notebook so the rendered page follows its teaching arc.

- Keep code visible when it teaches a core step: model specification, the API being demonstrated, or an analysis needed to understand the argument.
- Add `hide-input` to cell metadata tags for incidental plotting, formatting, and data manipulation. Keep that code visible when the notebook is teaching those operations.
- Keep outputs visible when they support the explanation, including figures, comparisons, and relevant diagnostics. Add `hide-output` to cell metadata tags for trivial or non-essential outputs, such as MCMC sampling messages and progress displays, unless sampler behavior is central to the notebook's argument.
- Decide input and output visibility independently. A plotting cell can hide its code while showing its figure; a sampling cell can show its code while hiding routine output. Split mixed-purpose cells when needed to preserve useful output.
- Use visibility tags to reduce clutter, not to conceal warnings, unexpected results, or diagnostic problems that readers need to understand.

Add tags to each cell's `metadata.tags` list, preserving existing tags. Keep the code and stored outputs available in the notebook; hiding them changes their default presentation in the rendered documentation.

Check the rendered notebook when practical to confirm that the intended code and outputs are visible or collapsed. For metadata-only edits, validate the notebook structure and confirm that source code and stored results are unchanged; full MCMC execution is unnecessary for visibility tags alone.
