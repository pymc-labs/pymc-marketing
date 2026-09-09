# Issue automation security boundary

`issue_automation.py` calls the Anthropic Messages API directly, with no tools,
Claude Code process, shell execution, repository instructions, or MCP servers.
Issue text is sent as data. Prompt wording is guidance, not the security boundary.

The analysis job has a read-only GitHub token. It produces only validated JSON:
allowlisted labels or up to three positive integer duplicate issue numbers.
The separate publishing job has issue-write permission, no Anthropic secret,
and validates the artifact again before making fixed GitHub REST calls. Neither
job interpolates issue content or model output into shell code. Checkouts use the
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

Run the offline regression suite with:

```sh
python -m unittest discover -s .github/scripts -p test_issue_automation.py -v
```

Deployment requires merging the workflow changes. Rotate the previously exposed
Anthropic credential and review historical workflow/API activity separately.
The tests do not establish whether previous runs were compromised. No test
sends live credentials or malicious payloads to a live GitHub workflow.
