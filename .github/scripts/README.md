# Issue automation security boundary

`issue_automation.py` calls the Anthropic Messages API directly, with no tools,
Claude Code process, shell execution, repository instructions, or MCP servers.
Issue text is sent as data. Prompt wording is guidance, not the security boundary.

The analysis job has a read-only GitHub token. It produces only validated JSON:
allowlisted labels or up to three positive integer duplicate issue numbers.
The separate publishing job has issue-write permission, no Anthropic secret,
and validates the artifact again before making fixed GitHub REST calls. Neither
job interpolates issue content or model output into shell code. Issue checkouts use the
triggering default-branch commit and do not persist credentials.

Duplicate retrieval is intentionally bounded to the 200 most recently updated
open issues, with 2,000 body characters per candidate. The current issue body is
limited to 20,000 characters. Older issues and details beyond these limits can be
missed. Duplicate publication checks that references are open issues in this
repository. Only this workflow's bot-authored marker comments are updated.

Free-form model reasoning is no longer posted or uploaded. Labels and fixed
numeric duplicate links remain; malformed, truncated, or tool-use responses
fail closed without publication. The model can still misclassify issues, but
cannot execute commands or invent labels that trigger other automations.

The PR labeler keeps its draft, external-repository and bot exclusions (including
its Copilot exception). It executes the helper from the trusted PR base SHA,
never the PR head or merge checkout. Changed files and bounded patch excerpts
are fetched through the GitHub API and passed to the model as JSON data.
File retrieval is capped at GitHub's 3,000-file limit; patches have a 2,000-character
per-file and 80,000-character total budget. Missing or truncated patches are
explicitly identified as excerpts in the prompt. A changed head revision,
closed PR or newly ineligible PR fails before publication.

PR output must contain exactly one approved release-note category and optional
approved module labels. The bot posts a fixed sticky summary of those labels.
No arbitrary model prose is published. The workflow tests verify analysis/write
job isolation and the trusted checkout refs for all three workflows.

All seven former Claude workflows were disabled operationally during incident
mitigation. After merge and credential rotation, re-enable only Triage, Duplicate
Detection, and PR Release Notes Labeler. Research, Plan, Iterate and Implement
are retired by this PR and should remain disabled. Existing workflow runs use
their original code: do not rerun historical runs to verify the fix.

Run the offline regression suite with:

```sh
uv run --no-project --with PyYAML==6.0.3 python -m unittest discover -s .github/scripts -p test_issue_automation.py -v
```

Deployment requires merging the workflow changes. Rotate the previously exposed
Anthropic credential and review historical workflow/API activity separately.
The tests do not establish whether previous runs were compromised. No test
sends live credentials or malicious payloads to a live GitHub workflow.
