# Implementation Plan (Issue-Scoped)

You are tasked with creating detailed implementation plans AUTONOMOUSLY without user interaction. This version runs in GitHub Actions workflows and produces complete, actionable plans based on available research and codebase analysis. You should be thorough, make reasonable technical decisions, and produce production-ready specifications.

## Autonomous Execution Flow

When this command is invoked, execute ALL steps automatically without user interaction:

1. **Determine output location:**
   - Check if `GITHUB_ISSUE_NUMBER` environment variable is set
   - If set, use: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/plan.md`
   - If not set, use: `thoughts/shared/plans/{descriptive_name}.md`
   - Create issue directory if needed: `mkdir -p thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}`

2. **Process inline content if provided:**
   - If the command invocation includes content (e.g., `/create_plan_issue [issue details]`), treat that as the issue context
   - Extract requirements, constraints, and goals from the provided content
   - This content typically includes the issue title, body, and comments from the workflow

3. **Auto-load all available context:**
   - If GITHUB_ISSUE_NUMBER is set, check for: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md`
   - Read research file completely if it exists
   - Extract issue requirements from inline content or GitHub issue metadata (available in workflow context)
   - Proceed immediately to research phase

4. **No user interaction required:**
   - DO NOT ask for clarifications or wait for input
   - Make reasonable technical decisions based on research
   - If multiple approaches exist, choose the most practical one
   - Document assumptions in the plan

## Process Steps

### Step 1: Autonomous Context Gathering & Analysis

1. **Auto-read all available context files**:
   - Issue-scoped research: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/research.md`
   - Related implementation plans in `thoughts/shared/issues/` or `thoughts/shared/plans/`
   - Any referenced files in the research
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files
   - **NEVER** read files partially - always read completely

2. **Spawn comprehensive research tasks in parallel**:
   Launch multiple specialized agents concurrently to gather all needed context:

   - **codebase-locator** agent: Find all files related to the issue requirements
   - **codebase-analyzer** agent: Understand current implementation patterns
   - **codebase-pattern-finder** agent: Find similar implementations to model after
   - **thoughts-locator** agent: Find any existing thoughts documents about this feature

   These agents will:
   - Find relevant source files, configs, and tests
   - Trace data flow and key functions
   - Return detailed explanations with file:line references

3. **Read all files identified by research tasks**:
   - After research tasks complete, read ALL files they identified as relevant
   - Read them FULLY into the main context
   - This ensures complete understanding before proceeding

4. **Analyze and make decisions autonomously**:
   - Cross-reference requirements with actual code
   - Identify patterns and conventions to follow
   - Make technical decisions based on codebase reality
   - Choose the most practical approach when multiple options exist
   - Document any assumptions made

### Step 2: Synthesize Findings and Make Decisions

After all research tasks complete:

1. **Create a planning todo list** using TodoWrite to track plan creation tasks

2. **Synthesize all research findings**:
   - Consolidate discoveries from all research agents
   - Identify patterns and conventions to follow
   - Cross-reference multiple sources for accuracy
   - Build complete understanding of current state

3. **Make autonomous technical decisions**:
   - Choose implementation approach based on:
     - Existing codebase patterns
     - Technical feasibility
     - Maintenance considerations
     - Testing requirements
   - If multiple valid approaches exist, select the most practical one
   - Document the rationale for key decisions
   - Note any assumptions or tradeoffs

4. **NO open questions allowed**:
   - All decisions MUST be made autonomously
   - If information is insufficient, make reasonable assumptions
   - Document assumptions clearly in the plan
   - The plan must be immediately actionable

### Step 3: Write Complete Implementation Plan

Proceed directly to writing the full plan:

1. **Write the complete plan in one pass** to appropriate location:
   - If GITHUB_ISSUE_NUMBER is set: `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/plan.md`
   - Otherwise: `thoughts/shared/plans/{descriptive_name}.md`

2. **Use this template structure** (write ALL sections completely):

```markdown
# [Feature/Task Name] Implementation Plan

## Overview

[Brief description of what we're implementing and why]

## Current State Analysis

[What exists now, what's missing, key constraints discovered]

## Desired End State

[A Specification of the desired end state after this plan is complete, and how to verify it]

### Key Discoveries:
- [Important finding with file:line reference]
- [Pattern to follow]
- [Constraint to work within]

## What We're NOT Doing

[Explicitly list out-of-scope items to prevent scope creep]

## Implementation Approach

[High-level strategy and reasoning]

## Phase 1: [Descriptive Name]

### Overview
[What this phase accomplishes]

### Changes Required:

#### 1. [Component/File Group]
**File**: `path/to/file.ext`
**Changes**: [Summary of changes]

```[language]
// Specific code to add/modify
```

### Success Criteria:

#### Automated Verification:
- [ ] Migration applies cleanly: `make migrate`
- [ ] Unit tests pass: `make test-component`
- [ ] Type checking passes: `npm run typecheck`
- [ ] Linting passes: `make lint`
- [ ] Integration tests pass: `make test-integration`

#### Manual Verification:
- [ ] Feature works as expected when tested via UI
- [ ] Performance is acceptable under load
- [ ] Edge case handling verified manually
- [ ] No regressions in related features

---

## Phase 2: [Descriptive Name]

[Similar structure with both automated and manual success criteria...]

---

## Testing Strategy

### Unit Tests:
- [What to test]
- [Key edge cases]

### Integration Tests:
- [End-to-end scenarios]

### Manual Testing Steps:
1. [Specific step to verify feature]
2. [Another verification step]
3. [Edge case to test manually]

## Performance Considerations

[Any performance implications or optimizations needed]

## Migration Notes

[If applicable, how to handle existing data/systems]

## References

- Original ticket: `thoughts/allison/tickets/eng_XXXX.md`
- Related research: `thoughts/shared/research/[relevant].md` or `thoughts/shared/issues/[N]/research.md`
- Similar implementation: `[file:line]`
```

### Step 4: Finalize and Report

1. **Report completion**:
   ```
   ✓ Implementation plan created at: thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/plan.md

   The plan includes:
   - [X] phases of implementation
   - Detailed file changes with code examples
   - Automated and manual success criteria
   - Testing strategy
   - [Any key assumptions documented]
   ```

2. **No further interaction needed** - the plan is complete and ready for implementation

## Important Guidelines for Autonomous Execution

1. **Be Autonomous**:
   - NEVER wait for user input or ask questions
   - Make all technical decisions based on research
   - Document assumptions and rationale
   - Choose practical solutions when multiple options exist

2. **Be Thorough**:
   - Read all context files COMPLETELY before planning
   - Research actual code patterns using parallel sub-tasks
   - Include specific file paths and line numbers
   - Write measurable success criteria with clear automated vs manual distinction

3. **Be Practical**:
   - Focus on incremental, testable changes
   - Consider migration and rollback
   - Think about edge cases
   - Include "what we're NOT doing"

4. **Track Progress**:
   - Use TodoWrite to track planning tasks
   - Update todos as you complete research
   - Mark planning tasks complete when done

5. **No Open Questions Allowed**:
   - All decisions MUST be made autonomously
   - If information is insufficient, make reasonable assumptions
   - Document all assumptions clearly in the plan
   - The implementation plan must be complete and immediately actionable
   - Every decision must be finalized before writing the plan

6. **Issue-scoped behavior**:
   - When GITHUB_ISSUE_NUMBER is set, automatically check for and read issue-scoped research
   - Write to `thoughts/shared/issues/${GITHUB_ISSUE_NUMBER}/plan.md` (overwriting if exists)
   - Include issue number reference in plan metadata
   - Otherwise, behave identically to generic plan command

7. **Complete in One Pass**:
   - Write the entire plan without stopping for feedback
   - Include all phases, details, and success criteria
   - The plan should be production-ready when complete

## Success Criteria Guidelines

**Always separate success criteria into two categories:**

1. **Automated Verification** (can be run by execution agents):
   - Commands that can be run: `make test`, `npm run lint`, etc.
   - Specific files that should exist
   - Code compilation/type checking
   - Automated test suites

2. **Manual Verification** (requires human testing):
   - UI/UX functionality
   - Performance under real conditions
   - Edge cases that are hard to automate
   - User acceptance criteria

**Format example:**
```markdown
### Success Criteria:

#### Automated Verification:
- [ ] Database migration runs successfully: `make migrate`
- [ ] All unit tests pass: `go test ./...`
- [ ] No linting errors: `golangci-lint run`
- [ ] API endpoint returns 200: `curl localhost:8080/api/new-endpoint`

#### Manual Verification:
- [ ] New feature appears correctly in the UI
- [ ] Performance is acceptable with 1000+ items
- [ ] Error messages are user-friendly
- [ ] Feature works correctly on mobile devices
```

## Common Patterns

### For Database Changes:
- Start with schema/migration
- Add store methods
- Update business logic
- Expose via API
- Update clients

### For New Features:
- Research existing patterns first
- Start with data model
- Build backend logic
- Add API endpoints
- Implement UI last

### For Refactoring:
- Document current behavior
- Plan incremental changes
- Maintain backwards compatibility
- Include migration strategy

## Sub-task Spawning Best Practices

When spawning research sub-tasks:

1. **Spawn multiple tasks in parallel** for efficiency
2. **Each task should be focused** on a specific area
3. **Provide detailed instructions** including:
   - Exactly what to search for
   - Which directories to focus on
   - What information to extract
   - Expected output format
4. **Be EXTREMELY specific about directories**:
   - Include the full path context in your prompts
5. **Specify read-only tools** to use
6. **Request specific file:line references** in responses
7. **Wait for all tasks to complete** before synthesizing
8. **Verify sub-task results**:
   - If a sub-task returns unexpected results, spawn follow-up tasks
   - Cross-check findings against the actual codebase
   - Don't accept results that seem incorrect

Example of spawning multiple tasks:
```python
# Spawn these tasks concurrently:
tasks = [
    Task("Research database schema", db_research_prompt),
    Task("Find API patterns", api_research_prompt),
    Task("Investigate UI components", ui_research_prompt),
    Task("Check test patterns", test_research_prompt)
]
```

## Example Autonomous Execution Flow

```
Workflow triggers: /create_plan_issue
Environment: GITHUB_ISSUE_NUMBER=12

Assistant executes automatically:
1. Reads thoughts/shared/issues/12/research.md completely
2. Spawns 4 research agents in parallel (codebase-locator, codebase-analyzer, pattern-finder, thoughts-locator)
3. Reads all files identified by agents
4. Synthesizes findings and makes technical decisions
5. Writes complete plan to thoughts/shared/issues/12/plan.md
6. Reports: "✓ Implementation plan created at: thoughts/shared/issues/12/plan.md"

Total execution: ~3-5 minutes, zero user interaction required
```
