# Plan Post-Implementation Analysis

You are tasked with analyzing what diverged between a TDD implementation plan and the actual implementation, then appending insights to help improve future planning.

## Initial Response

When this command is invoked:

1. **Check if a plan path was provided**:
   - If a plan file path was provided as a parameter, read it immediately
   - If no path provided, respond with:
   ```
   I'll analyze what diverged between your TDD plan and actual implementation.

   Please provide:
   1. Path to the TDD plan file (e.g., thoughts/shared/plans/feature_name_tdd.md)
   2. (Optional) Specific areas you want me to focus on

   I'll compare the plan against the actual implementation and append findings to help you improve future planning.

   Tip: You can invoke this command with the plan directly: `/plan_postmortem thoughts/shared/plans/my_feature_tdd.md`
   ```
   Then wait for user input.

## Process Steps

### Step 1: Read and Understand the Plan

1. **Read the plan file FULLY**:
   - Use Read tool WITHOUT limit/offset to read entire plan
   - Identify all phases and success criteria
   - Extract all file references mentioned in the plan
   - Note expected test files and implementation files
   - Understand the planned implementation order

2. **Extract key metadata**:
   - When was the plan created? (check git log for the plan file)
   - What files were supposed to be created/modified?
   - What were the test expectations?
   - What were the success criteria?

### Step 2: Analyze Actual Implementation

1. **Create research todo list** using TodoWrite to track analysis tasks

2. **Get git history context**:
   - Find when the plan was created: `git log --follow --format="%H %ai" thoughts/shared/plans/[plan_file] | tail -1`
   - Get all commits since plan creation that touched relevant files
   - Identify which commits were fixes vs. initial implementation

3. **Spawn parallel research tasks**:

   Use **codebase-analyzer** agents to:
   - Analyze each implementation file mentioned in the plan
   - Understand the actual implementation vs. planned implementation
   - Look for patterns that suggest fixes or workarounds

   Use **codebase-locator** agents to:
   - Find all test files related to this feature
   - Find any additional implementation files not in the plan
   - Locate any helper/utility files that were created

   Use **codebase-pattern-finder** agents to:
   - Find similar bug patterns in git history
   - Look for common issues in related features

4. **Wait for ALL sub-tasks to complete** before proceeding

5. **Run git analysis commands**:
   ```bash
   # Get all commits affecting the feature files since plan creation
   git log --since="[plan_creation_date]" --oneline -- [files_from_plan]

   # Get detailed diffs for fix commits (look for commit messages with "fix", "bug", etc.)
   git log --since="[plan_creation_date]" --grep="fix\|bug\|error" --oneline -- [files]
   ```

### Step 3: Identify Divergences

Analyze and categorize what diverged:

1. **Test Divergences**:
   - Tests that failed differently than expected
   - Additional tests needed that weren't in plan
   - Tests that were removed or skipped
   - Test structure changes

2. **Implementation Divergences**:
   - Files not mentioned in plan that were needed
   - Different implementation approach than planned
   - Additional dependencies or utilities needed
   - Order of implementation different from plan

3. **Bug Fixes & Issues**:
   - What bugs were encountered?
   - Which commits fixed issues?
   - What error patterns emerged?
   - What edge cases weren't anticipated?

4. **Success Criteria**:
   - Which automated checks failed initially?
   - Which manual verifications revealed issues?
   - Were there success criteria not in the plan?

### Step 4: Extract Lessons

Synthesize findings into actionable lessons:

1. **Planning Gaps**:
   - What should have been researched more deeply?
   - What assumptions were wrong?
   - What dependencies were missed?

2. **Test Design Issues**:
   - Were expected failure modes accurate?
   - Were edge cases identified?
   - Was test data realistic?

3. **Implementation Insights**:
   - Was the implementation order optimal?
   - Were there circular dependencies?
   - Were there integration issues?

4. **Future Improvements**:
   - What would make the next plan better?
   - What patterns should be documented?
   - What should be researched before planning?

### Step 5: Append Post-Implementation Analysis

**CRITICAL**: ONLY APPEND to the plan file, NEVER modify existing content.

1. **Use the Edit tool to append** the following section at the end of the plan:

```markdown

---

## Post-Implementation Analysis

**Date**: [Current date and time]
**Analyzed by**: [Researcher name from git config]
**Implementation Period**: [Plan creation date] to [Last relevant commit date]
**Relevant Commits**: [List of commit hashes that were part of implementation]

### What Worked As Planned

- [Success 1 with reference to plan phase]
- [Success 2 with reference to plan phase]
- [Aspect that followed the plan closely]

### Divergences from Plan

#### Tests

**Issue**: [What was different about the tests]
- **Planned**: [What the plan said]
- **Actual**: [What actually happened]
- **Files**: `file.py:line` - [description]
- **Commits**: `abc1234` - [commit message]
- **Why**: [Root cause of divergence]

[Repeat for each test divergence]

#### Implementation

**Issue**: [What was different about implementation]
- **Planned**: [What the plan expected]
- **Actual**: [What was actually done]
- **Files**: `file.py:line` - [description]
- **Commits**: `abc1234` - [commit message]
- **Why**: [Root cause of divergence]

[Repeat for each implementation divergence]

#### Additional Changes

- `file.py` - [File not in plan, why it was needed]
- `another.py` - [Unexpected modification, reason]

### Bugs and Fixes Encountered

#### Bug: [Short description]
- **Symptom**: [How it manifested]
- **Root Cause**: [What was wrong]
- **Fix**: [What was changed] in `file.py:line`
- **Commit**: `abc1234` - [commit message]
- **Plan Gap**: [What could have prevented this]

[Repeat for each significant bug]

### Success Criteria Gaps

#### Automated Checks
- [ ] [Check from plan] - [PASSED/FAILED/NOT RUN]
- [ ] [Additional check needed] - [Why it was needed]

#### Manual Verification
- [ ] [Verification from plan] - [Result]
- [ ] [Additional verification] - [Why it was needed]

### Lessons Learned

#### For Future Planning

1. **Research More Deeply**: [Specific area that needed more research]
   - Next time: [Actionable improvement]

2. **Assumptions to Challenge**: [Wrong assumption made]
   - Next time: [How to validate this]

3. **Dependencies to Check**: [Missed dependency]
   - Next time: [How to discover this earlier]

[Continue with other lessons]

#### For Test Design

1. **[Lesson about test design]**
   - Example: [Specific case from this implementation]
   - Next time: [Actionable improvement]

[Continue with other test design lessons]

#### For Implementation

1. **[Lesson about implementation]**
   - Example: [Specific case from this implementation]
   - Next time: [Actionable improvement]

[Continue with other implementation lessons]

### Recommendations for Next Similar Plan

1. **[Specific recommendation]** - [Why this would help]
2. **[Another recommendation]** - [Context and benefit]
3. **[Process improvement]** - [How to apply this]

### Patterns Worth Documenting

- **[Pattern 1]**: [Where it appeared, why it's worth documenting]
- **[Pattern 2]**: [Context and reference]

### Open Questions for Future Work

- [Question about the implementation that remains]
- [Potential improvement or refactoring needed]
- [Related feature to consider]

---

*This post-implementation analysis helps improve future TDD planning by documenting what actually happened vs. what was planned.*
```

### Step 6: Present Summary

After appending the analysis:

1. **Show the user a concise summary**:
   ```
   Post-implementation analysis added to: [plan_path]

   Key Findings:
   - [# of tests] test divergences
   - [# of bugs] bugs encountered and fixed
   - [# of files] additional files needed
   - [# of lessons] lessons learned for future planning

   Top Lessons:
   1. [Most important lesson]
   2. [Second important lesson]
   3. [Third important lesson]

   The full analysis with git references, file references, and detailed recommendations has been appended to the plan.
   ```

2. **Ask follow-up questions**:
   ```
   Would you like me to:
   - Dive deeper into any specific divergence?
   - Create a checklist for your next plan based on these lessons?
   - Analyze patterns across multiple plans?
   ```

## Important Guidelines

1. **Only Append, Never Modify**:
   - Use Edit tool to append at the end of the file
   - Never change existing plan content
   - The plan is a historical record; we're adding a retrospective

2. **Be Specific and Actionable**:
   - Include actual file:line references
   - Include commit hashes for traceability
   - Make lessons concrete and actionable
   - Focus on "next time" improvements

3. **Focus on Learning, Not Blame**:
   - Frame divergences as learning opportunities
   - Identify what was hard to predict vs. what was missed
   - Suggest improvements, don't criticize the original plan
   - Acknowledge what worked well

4. **Git Integration**:
   - Use git log to understand implementation timeline
   - Reference specific commits for divergences
   - Look at commit messages for clues about issues
   - Check diffs to understand what actually changed

5. **Balance Detail and Brevity**:
   - Be thorough on significant divergences
   - Summarize minor issues
   - Focus on patterns over individual details
   - Make it skimmable with clear headers

6. **Make It Actionable**:
   - Every lesson should have a "next time" action
   - Recommendations should be concrete
   - Patterns should be clearly defined
   - Questions should be specific

7. **Use TodoWrite**:
   - Track analysis tasks
   - Mark tasks complete as you finish each section
   - Show progress to user

## Example Divergence Categories

**Common Test Divergences**:
- Tests failed with different errors than expected
- Additional edge cases discovered during implementation
- Test fixtures needed refactoring
- Mock setup more complex than anticipated

**Common Implementation Divergences**:
- Circular import issues required restructuring
- Additional utility functions needed
- Different algorithm required for performance
- External API behaved differently than documented

**Common Bug Patterns**:
- Off-by-one errors
- Null/None handling
- Type coercion issues
- Race conditions or timing issues
- State management problems

**Common Planning Gaps**:
- Didn't research existing similar implementations
- Didn't verify external API behavior
- Didn't check for circular dependencies
- Didn't consider database transaction boundaries
- Didn't account for async/await implications

## Success Metrics

A good post-implementation analysis should:

- [ ] Be appended to the original plan (not modify it)
- [ ] Include specific git commit references
- [ ] Include specific file:line references
- [ ] Identify 3-5 key lessons learned
- [ ] Provide actionable "next time" improvements
- [ ] Document patterns worth remembering
- [ ] Be honest about what worked and what didn't
- [ ] Take 5-10 minutes to read and understand
- [ ] Help improve the next plan's quality
