# Test-Driven Development Implementation Plan

You are tasked with creating detailed TDD implementation plans through an interactive, iterative process. The core principle: **Write tests first, verify they fail properly, then implement features by debugging the failing tests.**

## Initial Response

When this command is invoked:

1. **Check if parameters were provided**:
   - If a file path or ticket reference was provided as a parameter, skip the default message
   - Immediately read any provided files FULLY
   - Begin the research process

2. **If no parameters provided**, respond with:
```
I'll help you create a Test-Driven Development implementation plan. We'll design comprehensive tests first, verify they fail properly, then implement features by making those tests pass.

Please provide:
1. The task/ticket description (or reference to a ticket file)
2. Any relevant context, constraints, or specific requirements
3. Links to related research or previous implementations
4. Specification/documentation for expected behavior

I'll analyze this information and work with you to create a test-first plan.

Tip: You can also invoke this command with a ticket file directly: `/create_plan_tdd thoughts/shared/tickets/feature_123.md`
```

Then wait for the user's input.

## Process Steps

### Step 1: Context Gathering & Initial Analysis

1. **Read all mentioned files immediately and FULLY**:
   - Ticket files
   - Specification documents (especially important for TDD!)
   - Research documents
   - Related implementation plans
   - Example test files
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files
   - **CRITICAL**: DO NOT spawn sub-tasks before reading these files yourself in the main context

2. **Spawn initial research tasks to gather context**:
   Before asking the user any questions, use specialized agents to research in parallel:

   - Use **codebase-locator** to find:
     - Existing test files and patterns
     - Similar features with tests
     - Test utilities and fixtures

   - Use **codebase-analyzer** to understand:
     - Current testing framework setup
     - Test patterns and conventions
     - Mock/fixture patterns used

   - Use **codebase-pattern-finder** to find:
     - Similar test implementations to model after
     - Test assertion patterns
     - Test organization structures

   - If relevant, use **thoughts-locator** to find existing research about testing approaches

3. **Read all files identified by research tasks**:
   - After research tasks complete, read ALL files they identified as relevant
   - Read them FULLY into the main context
   - Pay special attention to existing test patterns

4. **Analyze testing landscape**:
   - Understand current test framework (pytest, unittest, etc.)
   - Identify test utilities and helpers available
   - Note testing conventions to follow
   - Find fixture patterns and mock strategies

5. **Present informed understanding and focused questions**:
   ```
   Based on the ticket and my research, I understand we need to [accurate summary].

   I've found that:
   - [Current test patterns with file:line reference]
   - [Testing framework and utilities available]
   - [Similar tests we can model after]

   For TDD, I need to understand:
   - [Expected behavior/specification questions]
   - [Edge cases to test]
   - [Test data requirements]

   Questions that my research couldn't answer:
   - [Specific behavioral question]
   - [Edge case handling preference]
   ```

### Step 2: Test Design & Discovery

After getting initial clarifications:

1. **Create a research todo list** using TodoWrite to track test design and implementation tasks

2. **Spawn parallel sub-tasks for test research**:
   - Use **codebase-pattern-finder** to find similar test implementations
   - Use **codebase-analyzer** to understand test utilities and fixtures
   - Use **codebase-locator** to find test data or example inputs
   - If specs exist, use appropriate agents to research specification details

3. **Wait for ALL sub-tasks to complete** before proceeding

4. **Present test design options**:
   ```
   Based on my research, here's my test design strategy:

   **Test Categories:**
   1. [Category] - [what it validates]
   2. [Category] - [what it validates]

   **Key Test Cases:**
   - [Normal operation test]
   - [Edge case 1]
   - [Edge case 2]
   - [Error condition test]

   **Test Data Strategy:**
   - [How we'll generate/source test data]

   **Assertion Strategy:**
   - [What we'll verify and how]

   Does this test coverage look comprehensive?
   Are there edge cases I'm missing?
   ```

### Step 3: Plan Structure Development

Once aligned on test design:

1. **Create initial plan outline**:
   ```
   Here's my proposed TDD plan structure:

   ## Overview
   [1-2 sentence summary]

   ## TDD Phases:
   1. Test Design & Implementation - [what tests we'll write]
   2. Test Failure Verification - [verify tests fail correctly]
   3. Feature Implementation (Red → Green) - [implement to pass tests]
   4. Refactoring & Cleanup - [clean up while keeping tests green]

   Does this phasing make sense? Should I adjust the test categories?
   ```

2. **Get feedback on structure** before writing details

### Step 4: Detailed TDD Plan Writing

After structure approval:

1. **Write the plan** to `thoughts/shared/plans/{descriptive_name}_tdd.md`
2. **Use this template structure**:

```markdown
# [Feature/Task Name] TDD Implementation Plan

## Overview

[Brief description of what we're implementing and why, with TDD approach]

## Current State Analysis

[What exists now, what's missing, key constraints discovered]

### Current Testing Landscape:
- Testing framework: [pytest/unittest/etc]
- Available test utilities: [file:line references]
- Existing test patterns to follow: [examples]
- Test fixtures/mocks available: [what we can reuse]

## Desired End State

[Specification of desired behavior after implementation, defined by passing tests]

### Key Discoveries:
- [Important finding with file:line reference]
- [Test pattern to follow]
- [Testing constraint to work within]

## What We're NOT Testing/Implementing

[Explicitly list out-of-scope items to prevent scope creep]

## TDD Approach

[High-level testing and implementation strategy]

### Test Design Philosophy:
- [What makes a good test for this feature]
- [How we'll ensure tests are informative]
- [How we'll make failure messages diagnostic]

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive, informative tests that define the feature completely. These tests should fail in expected, diagnostic ways.

### Test Categories:

#### 1. [Test Category Name] (e.g., "Basic Operation Tests")
**Test File**: `tests/path/to/test_file.py`
**Purpose**: [What aspect of behavior these tests validate]

**Test Cases to Write:**

##### Test: `test_[specific_behavior]`
**Purpose**: [What this specific test validates]
**Test Data**: [Input data/fixtures needed]
**Expected Behavior**: [What should happen]
**Assertions**: [What to verify]

```python
def test_specific_behavior():
    """
    Test that [feature] correctly [does something].

    This test verifies:
    - [Specific behavior 1]
    - [Specific behavior 2]
    """
    # Arrange
    input_data = [test input]
    expected_output = [expected result]

    # Act
    actual_output = feature_function(input_data)

    # Assert
    assert actual_output == expected_output, \
        f"Expected {expected_output}, got {actual_output}"
    # Additional assertions...
```

**Expected Failure Mode**: [How this test should fail before implementation]
- Error type: [e.g., AttributeError, NotImplementedError]
- Expected message: [What error message we expect]

##### Test: `test_[edge_case_behavior]`
[Similar structure for each test case...]

#### 2. [Another Test Category]
[Similar structure...]

### Test Implementation Steps:

1. **Create/modify test file**: `tests/path/to/test_file.py`
2. **Import necessary testing utilities**:
   ```python
   import pytest
   from unittest.mock import Mock, patch
   # Other imports...
   ```

3. **Create test fixtures if needed**:
   ```python
   @pytest.fixture
   def sample_data():
       """Fixture providing test data."""
       return [test data]
   ```

4. **Implement each test case** (see test cases above)

5. **Add test documentation**: Ensure each test has clear docstrings explaining what it validates

### Success Criteria:

#### Automated Verification:
- [ ] All test files created with proper structure
- [ ] Tests use existing test utilities correctly
- [ ] Test code follows project conventions: `make lint-tests`
- [ ] Tests are discoverable: `pytest --collect-only tests/path/`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertion messages are diagnostic
- [ ] Test code is readable and maintainable

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in the expected, diagnostic ways. This ensures our tests are actually testing something and will catch regressions.

### Verification Steps:

1. **Run the test suite**:
   ```bash
   pytest tests/path/to/test_file.py -v
   ```

2. **For each test, verify**:
   - Test fails (not passes or errors unexpectedly)
   - Failure message is informative
   - Failure points to the right location
   - Error type matches expectations

3. **Document failure modes**:
   Create a checklist of expected vs actual failure behavior

### Expected Failures:

For each test, document what we expect:

- **test_[name]**:
  - Expected: `NotImplementedError: [feature] not yet implemented`
  - Points to: [specific function/method]

- **test_[name]**:
  - Expected: `AssertionError: Expected [X], got None`
  - Shows: Clear diff of expected vs actual

### Success Criteria:

#### Automated Verification:
- [ ] All tests run and are discovered: `pytest --collect-only`
- [ ] All tests fail (none pass): `pytest tests/path/ --tb=short`
- [ ] No unexpected errors (import errors, syntax errors): `pytest tests/path/ --tb=line`

#### Manual Verification:
- [ ] Each test fails with expected error type
- [ ] Failure messages clearly indicate what's missing
- [ ] Failure messages would help during implementation
- [ ] Stack traces point to relevant code locations
- [ ] No cryptic or misleading error messages

### Adjustment Phase:

If tests don't fail properly:
- [ ] Fix tests that pass unexpectedly (too lenient)
- [ ] Fix tests with confusing error messages
- [ ] Fix tests that error instead of fail (missing imports, etc.)
- [ ] Improve assertion messages for clarity

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement the feature by making tests pass, one at a time. Work like you're debugging - let the test failures guide you to what needs to be implemented next.

### Implementation Strategy:

**Order of Implementation:**
1. Start with [simplest/most fundamental test]
2. Then [next logical test]
3. Continue in order of dependency/complexity

### Implementation Steps:

#### Implementation 1: Make `test_[first_test]` Pass

**Target Test**: `test_[first_test]`
**Current Failure**: [What the test currently shows]

**Changes Required:**

**File**: `path/to/implementation_file.py`
**Changes**: [Summary of what to add/modify]

```python
# Code to implement
def feature_function(input_data):
    """
    [Docstring describing the function]

    Args:
        input_data: [description]

    Returns:
        [description]
    """
    # Implementation guided by test
    pass
```

**Debugging Approach:**
1. Run the test: `pytest tests/path/to/test_file.py::test_[first_test] -v`
2. Read the failure message carefully
3. Implement just enough to address the failure
4. Re-run test
5. Repeat until test passes

**Success Criteria:**

##### Automated Verification:
- [ ] Target test passes: `pytest tests/path/to/test_file.py::test_[first_test] -v`
- [ ] Previously passing tests still pass (if any)
- [ ] No new linting errors: `make lint`
- [ ] Type checking passes (if applicable): `mypy path/to/file.py`

##### Manual Verification:
- [ ] Implementation is clean and understandable
- [ ] Code follows project conventions
- [ ] No obvious performance issues
- [ ] Comments explain complex logic

#### Implementation 2: Make `test_[second_test]` Pass

[Similar structure for each test...]

### Complete Feature Implementation:

Once all individual tests pass:

**Final Integration:**
- Run full test suite: `pytest tests/path/ -v`
- Check for interactions between test cases
- Verify no regressions in other tests

**Success Criteria:**

##### Automated Verification:
- [ ] All new tests pass: `pytest tests/path/to/test_file.py -v`
- [ ] No regressions in existing tests: `pytest`
- [ ] Code coverage meets requirements: `pytest --cov=module tests/`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make typecheck`

##### Manual Verification:
- [ ] Implementation handles all edge cases
- [ ] Code is maintainable and clear
- [ ] Performance is acceptable
- [ ] No obvious bugs or issues

---

## Phase 4: Refactoring & Cleanup

### Overview
Now that tests are green, refactor to improve code quality while keeping tests passing. Tests protect us during refactoring.

### Refactoring Targets:

1. **Code Duplication**:
   - [Identify repeated patterns]
   - [Extract common functions]

2. **Code Clarity**:
   - [Complex logic to simplify]
   - [Variable/function names to improve]

3. **Performance**:
   - [Potential optimizations]
   - [Unnecessary operations to remove]

4. **Test Quality**:
   - [Test code duplication to extract]
   - [Test fixtures to create/improve]

### Refactoring Steps:

1. **Ensure all tests pass before starting**: `pytest -v`

2. **For each refactoring**:
   - Make the change
   - Run tests: `pytest tests/path/ -v`
   - If tests pass, commit the change
   - If tests fail, revert and reconsider

3. **Focus areas**:
   - Extract helper functions
   - Improve naming
   - Add code comments where needed
   - Simplify complex logic
   - Remove dead code

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass: `pytest -v`
- [ ] Code coverage maintained or improved: `pytest --cov`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make typecheck`
- [ ] No performance regressions: [performance test command if applicable]

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity added
- [ ] Function/variable names are clear
- [ ] Comments explain "why" not "what"
- [ ] Code follows project idioms and patterns

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Normal operation paths: [percentage/description]
- [ ] Edge cases: [specific cases]
- [ ] Error conditions: [error scenarios]
- [ ] Integration points: [what we test together]

### Test Organization:
- Test files: [where tests live]
- Fixtures: [where fixtures are defined]
- Test utilities: [helper functions]
- Test data: [where test data lives]

### Running Tests:

```bash
# Run all tests for this feature
pytest tests/path/to/test_file.py -v

# Run specific test
pytest tests/path/to/test_file.py::test_name -v

# Run with coverage
pytest tests/path/ --cov=module --cov-report=term-missing

# Run with failure details
pytest tests/path/ -vv --tb=short
```

## Performance Considerations

[Any performance implications or optimizations needed]

### Performance Testing:
- [ ] [Specific performance test to write]
- [ ] [Performance benchmark to meet]

## Migration Notes

[If applicable, how to handle existing data/systems]

## References

- Original ticket: `thoughts/shared/tickets/[ticket].md`
- Related research: `thoughts/shared/research/[relevant].md`
- Specification: [link or file reference]
- Similar implementation: `[file:line]`
- Test patterns reference: `[file:line]`
```

### Step 5: Review and Iteration

1. **Present the draft plan location**:
   ```
   I've created the TDD implementation plan at:
   `thoughts/shared/plans/[filename]_tdd.md`

   Please review it and let me know:
   - Are the test cases comprehensive?
   - Are the expected failure modes clear?
   - Is the implementation order logical?
   - Are the success criteria specific enough?
   - Any edge cases I'm missing?
   ```

2. **Iterate based on feedback** - be ready to:
   - Add missing test cases
   - Clarify expected failure modes
   - Adjust implementation order
   - Add edge case tests
   - Improve test design

3. **Continue refining** until the user is satisfied

## Important TDD Guidelines

1. **Tests First, Always**:
   - Write ALL tests before any implementation
   - Make sure tests fail properly before implementing
   - Implementation is "debugging" the failing tests

2. **Informative Test Failures**:
   - Test names should describe what they test
   - Assertion messages should be diagnostic
   - Docstrings should explain the "why"
   - Failure output should guide implementation

3. **One Test at a Time**:
   - Implement to make one test pass
   - Verify it passes
   - Move to next test
   - Don't try to make multiple tests pass at once

4. **Keep Tests Green**:
   - Once a test passes, it should stay passing
   - Run full suite regularly
   - Don't break passing tests while implementing new ones

5. **Refactor Fearlessly**:
   - Tests protect you during refactoring
   - Make small refactoring changes
   - Run tests after each change
   - Revert if tests fail

6. **Test Quality Matters**:
   - Tests should be as maintainable as production code
   - Extract test fixtures and utilities
   - Keep tests DRY (Don't Repeat Yourself)
   - But prefer clarity over cleverness in tests

7. **Be Skeptical**:
   - Question if test really tests what it claims
   - Verify tests fail when they should
   - Check edge cases thoroughly
   - Don't assume - verify with failing tests

8. **Be Interactive**:
   - Get feedback on test design before writing
   - Confirm expected failure modes
   - Validate implementation order
   - Work collaboratively

9. **Track Progress**:
   - Use TodoWrite to track test writing and implementation
   - Mark tests as complete as they pass
   - Update todos as you discover new edge cases

10. **No Untested Code**:
    - Every feature must have tests written first
    - Every line should be justified by a test
    - If you can't think of a test, reconsider the feature

## TDD Anti-Patterns to Avoid

1. **Writing implementation before tests** - defeats the purpose
2. **Tests that always pass** - not testing anything
3. **Unclear failure messages** - slows down debugging
4. **Testing implementation details** - makes refactoring hard
5. **Skipping the "verify failure" step** - might have false positives
6. **Making multiple tests pass at once** - too big of steps
7. **Not running tests frequently** - lose the feedback loop
8. **Ignoring test quality** - creates maintenance burden

## Success Metrics

A good TDD plan should result in:

- [ ] Comprehensive test coverage written before implementation
- [ ] All tests fail properly before implementation starts
- [ ] Clear, diagnostic failure messages that guide implementation
- [ ] Implementation proceeds in logical, testable steps
- [ ] All tests pass at the end
- [ ] Code is well-structured (thanks to refactoring phase)
- [ ] High confidence that code works correctly
- [ ] Easy-to-maintain test suite for future changes
