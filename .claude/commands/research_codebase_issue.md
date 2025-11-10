# Research Codebase (Issue-Scoped)

You are tasked with conducting comprehensive research across the codebase to answer user questions by spawning parallel sub-agents and synthesizing their findings. This version automatically writes to issue-scoped directories when running in GitHub workflow context.

## Initial Setup:

When this command is invoked:

1. **Determine output location:**
   - Check if `GITHUB_ISSUE_NUMBER` environment variable is set
   - If set, use: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md`
   - If not set, use: `thoughts/shared/research/YYYY-MM-DD_HH-MM-SS_topic.md`
   - Create issue directory if needed: `mkdir -p thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}`

2. **Check if research content is provided inline:**
   - If the command invocation includes additional content (e.g., `/research_codebase_issue [content]`), treat that content as the research query
   - If content is provided, skip the ready message and proceed directly to research (step 1 of "Steps to follow after receiving the research query")
   - If no content is provided, show the ready message below and wait for input

3. **Respond with context-aware message (only if no inline content):**

If GITHUB_ISSUE_NUMBER is set, respond with:
```
I'm ready to research the codebase for issue #${GITHUB_ISSUE_NUMBER}.
Output will be written to: thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md

Please provide your research question or area of interest, and I'll analyze it thoroughly.
```

Otherwise, respond with:
```
I'm ready to research the codebase. Please provide your research question or area of interest, and I'll analyze it thoroughly by exploring relevant components and connections.
```

Then wait for the user's research query.

## Steps to follow after receiving the research query:

1. **Read any directly mentioned files first:**
   - If the user mentions specific files (tickets, docs, JSON), read them FULLY first
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files
   - **CRITICAL**: Read these files yourself in the main context before spawning any sub-tasks
   - This ensures you have full context before decomposing the research

2. **Analyze and decompose the research question:**
   - Break down the user's query into composable research areas
   - Take time to ultrathink about the underlying patterns, connections, and architectural implications the user might be seeking
   - Identify specific components, patterns, or concepts to investigate
   - Create a research plan using TodoWrite to track all subtasks
   - Consider which directories, files, or architectural patterns are relevant

3. **Spawn parallel sub-agent tasks for comprehensive research:**
   - Create multiple Task agents to research different aspects concurrently

   The key is to use these agents intelligently:
   - Start with locator agents to find what exists
   - Then use analyzer agents on the most promising findings
   - Run multiple agents in parallel when they're searching for different things
   - Each agent knows its job - just tell it what you're looking for
   - Don't write detailed prompts about HOW to search - the agents already know

4. **Wait for all sub-agents to complete and synthesize findings:**
   - IMPORTANT: Wait for ALL sub-agent tasks to complete before proceeding
   - Compile all sub-agent results (both codebase and thoughts findings)
   - Prioritize live codebase findings as primary source of truth
   - Use thoughts/ findings as supplementary historical context
   - Connect findings across different components
   - Include specific file paths and line numbers for reference
   - Verify all thoughts/ paths are correct (e.g., thoughts/allison/ not thoughts/shared/ for personal files)
   - Highlight patterns, connections, and architectural decisions
   - Answer the user's specific questions with concrete evidence

5. **Gather metadata for the research document:**
   - generate all relevant metadata
   - Determine filename based on environment:
     - If GITHUB_ISSUE_NUMBER is set: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md`
     - Otherwise: `thoughts/shared/research/YYYY-MM-DD_HH-MM-SS_topic.md`

6. **Generate research document:**
   - Use the metadata gathered in step 5
   - Structure the document with YAML frontmatter followed by content:
     ```markdown
     ---
     date: [Current date and time with timezone in ISO format]
     researcher: [Researcher name]
     git_commit: [Current commit hash]
     branch: [Current branch name]
     repository: [Repository name]
     topic: "[User's Question/Topic]"
     tags: [research, codebase, relevant-component-names]
     status: complete
     last_updated: [Current date in YYYY-MM-DD format]
     last_updated_by: [Researcher name]
     issue_number: [GITHUB_ISSUE_NUMBER if set, otherwise omit this field]
     ---

     # Research: [User's Question/Topic]

     **Date**: [Current date and time with timezone from step 5]
     **Researcher**: [Researcher name]
     **Git Commit**: [Current commit hash from step 5]
     **Branch**: [Current branch name from step 5]
     **Repository**: [Repository name]
     **Issue**: [#GITHUB_ISSUE_NUMBER if set, otherwise omit this line]

     ## Research Question
     [Original user query]

     ## Summary
     [High-level findings answering the user's question]

     ## Detailed Findings

     ### [Component/Area 1]
     - Finding with reference ([file.ext:line](link))
     - Connection to other components
     - Implementation details

     ### [Component/Area 2]
     ...

     ## Code References
     - `path/to/file.py:123` - Description of what's there
     - `another/file.ts:45-67` - Description of the code block

     ## Architecture Insights
     [Patterns, conventions, and design decisions discovered]

     ## Historical Context (from thoughts/)
     [Relevant insights from thoughts/ directory with references]
     - `thoughts/shared/something.md` - Historical decision about X
     - `thoughts/local/notes.md` - Past exploration of Y
     Note: Paths exclude "searchable/" even if found there

     ## Related Research
     [Links to other research documents in thoughts/shared/research/]

     ## Open Questions
     [Any areas that need further investigation]
     ```

7. **Add GitHub permalinks (if applicable):**
   - Check if on main branch or if commit is pushed: `git branch --show-current` and `git status`
   - If on main/master or pushed, generate GitHub permalinks:
     - Get repo info: `gh repo view --json owner,name`
     - Create permalinks: `https://github.com/{owner}/{repo}/blob/{commit}/{file}#L{line}`
   - Replace local file references with permalinks in the document

8. **Sync and present findings:**
   - Present a concise summary of findings to the user
   - Include key file references for easy navigation

## Important notes:
- Always use parallel Task agents to maximize efficiency and minimize context usage
- Always run fresh codebase research - never rely solely on existing research documents
- The thoughts/ directory provides historical context to supplement live findings
- Focus on finding concrete file paths and line numbers for developer reference
- Research documents should be self-contained with all necessary context
- Each sub-agent prompt should be specific and focused on read-only operations
- Consider cross-component connections and architectural patterns
- Include temporal context (when the research was conducted)
- Link to GitHub when possible for permanent references
- Keep the main agent focused on synthesis, not deep file reading
- Encourage sub-agents to find examples and usage patterns, not just definitions
- Explore all of thoughts/ directory, not just research subdirectory
- **File reading**: Always read mentioned files FULLY (no limit/offset) before spawning sub-tasks
- **Critical ordering**: Follow the numbered steps exactly
  - ALWAYS read mentioned files first before spawning sub-tasks (step 1)
  - ALWAYS wait for all sub-agents to complete before synthesizing (step 4)
  - ALWAYS gather metadata before writing the document (step 5 before step 6)
  - NEVER write the research document with placeholder values
- **Path handling**: The thoughts/searchable/ directory contains hard links for searching
  - Always document paths by removing ONLY "searchable/" - preserve all other subdirectories
  - Examples of correct transformations:
    - `thoughts/searchable/allison/old_stuff/notes.md` → `thoughts/allison/old_stuff/notes.md`
    - `thoughts/searchable/shared/prs/123.md` → `thoughts/shared/prs/123.md`
    - `thoughts/searchable/global/shared/templates.md` → `thoughts/global/shared/templates.md`
  - NEVER change allison/ to shared/ or vice versa - preserve the exact directory structure
  - This ensures paths are correct for editing and navigation
- **Frontmatter consistency**:
  - Always include frontmatter at the beginning of research documents
  - Keep frontmatter fields consistent across all research documents
  - Update frontmatter when adding follow-up research
  - Use snake_case for multi-word field names (e.g., `last_updated`, `git_commit`)
  - Tags should be relevant to the research topic and components studied
  - Add `issue_number` field when GITHUB_ISSUE_NUMBER environment variable is set
- **Issue-scoped behavior**:
  - When GITHUB_ISSUE_NUMBER is set, create directory `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/` if needed
  - Write to `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md` (overwriting if exists)
  - Include issue number in frontmatter and document header
  - Otherwise, behave identically to generic research command
