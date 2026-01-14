# Verify Tests

You are tasked with validating the quality and correctness of tests written for a TDD implementation plan. This command ensures tests are complete, atomic, informative, and actually testing the right things (not compensating for implementation bugs).

## Initial Setup

When invoked:

1. **Check for uncommitted test files**:
   ```bash
   # Get list of modified/new test files
   git status --porcelain tests/
   ```
   - Filter for test files that are modified (M) or untracked (??)
   - If no uncommitted test files found, inform user and exit
   - These are the ONLY test files that will be verified

2. **Locate the TDD implementation plan**:
   - If plan path provided as parameter, use it
   - Otherwise, search `thoughts/shared/plans/*_tdd.md` for recent plans
   - Ask user if multiple candidates found

3. **Read the plan completely**:
   - Identify all test files that should exist
   - Extract all planned test cases from the plan
   - Note expected test behavior and assertions
   - **Filter plan expectations to only uncommitted test files identified in step 1**

4. **Gather test implementation evidence for uncommitted files only**:
   ```bash
   # Check test discovery for uncommitted test files only
   uv run pytest --collect-only <uncommitted_test_files>

   # Run uncommitted tests with verbose output
   uv run pytest <uncommitted_test_files> -vv
   ```

## Validation Process

### Step 1: Test Completeness Analysis

**Goal**: Verify all planned tests in uncommitted files were actually implemented.

1. **Extract planned tests from the plan** (for uncommitted test files only):
   - Parse all `test_*` function names from the plan that belong to uncommitted test files
   - Note which test categories they belong to
   - Track expected test file locations (only uncommitted ones)

2. **Discover implemented tests in uncommitted files**:
   ```bash
   # Collect tests from uncommitted files only
   pytest --collect-only <uncommitted_test_files> -q
   ```

3. **Compare planned vs implemented** (for uncommitted files):
   - Create a checklist of planned tests in uncommitted files
   - Mark which ones exist in the codebase
   - Identify missing tests
   - Identify extra tests (not in plan - may be good!)

4. **Read all uncommitted test files**:
   - Use Read tool to load each uncommitted test file completely
   - Don't use limit/offset - read entire files

**Success Criteria**:
- [ ] All planned test cases are implemented
- [ ] Test file structure matches plan
- [ ] No critical test categories are missing

### Step 2: Test Atomicity Analysis

**Goal**: Verify each test focuses on one specific behavior.

For each test function:

1. **Analyze test structure**:
   - Count assertions in the test
   - Check if multiple different behaviors are tested
   - Look for multiple unrelated arrange-act-assert cycles

2. **Check for atomic violations**:
   - ❌ Test checks multiple unrelated features
   - ❌ Test has multiple independent assertion groups
   - ❌ Test name uses "and" suggesting multiple behaviors
   - ❌ Test would require multiple different fixes if it failed

3. **Evaluate test focus**:
   ```
   Good (Atomic):
   def test_add_returns_sum_of_two_numbers():
       result = add(2, 3)
       assert result == 5

   Bad (Not Atomic):
   def test_calculator_operations():
       assert add(2, 3) == 5
       assert subtract(5, 2) == 3
       assert multiply(2, 3) == 6  # Three different features
   ```

**Success Criteria**:
- [ ] Each test focuses on one behavior
- [ ] Test names describe a single expectation
- [ ] A failing test points to one specific issue

### Step 3: Test Informativeness Analysis

**Goal**: Verify tests provide clear, diagnostic information when they fail.

For each test:

1. **Check test naming**:
   - Does name clearly describe what is being tested?
   - Is it obvious what behavior is expected?
   - Would a failing test name help locate the bug?

2. **Evaluate docstrings**:
   ```python
   # Good docstring
   def test_division_by_zero_raises_value_error():
       """
       Test that dividing by zero raises ValueError with clear message.

       This ensures users get informative errors rather than
       cryptic ZeroDivisionError messages.
       """

   # Bad docstring (or missing)
   def test_division():
       # No docstring explaining why this test exists
   ```

3. **Analyze assertion messages**:
   ```python
   # Good - informative
   assert result == expected, \
       f"Division failed: {numerator}/{denominator} returned {result}, expected {expected}"

   # Bad - not informative
   assert result == expected  # No message
   ```

4. **Check failure diagnostics**:
   - Run tests and examine failure output
   - Are failure messages clear?
   - Do they show what was expected vs actual?
   - Do they provide context for debugging?

**Success Criteria**:
- [ ] Test names clearly describe behavior
- [ ] Tests have informative docstrings explaining "why"
- [ ] Assertion messages are diagnostic
- [ ] Failure output would help locate bugs

### Step 4: Implementation Compensation Analysis

**Goal**: Ensure tests aren't hiding bugs or testing the wrong things.

This is the most critical and nuanced validation. Tests should validate correct behavior, not work around implementation bugs.

#### 4.1: Check for "Tests That Pass for Wrong Reasons"

1. **Look for suspicious patterns**:
   ```python
   # Suspicious: Test might be too lenient
   def test_parse_date():
       result = parse_date("2024-01-32")  # Invalid date!
       assert result is not None  # Just checks it returns something

   # Better: Test validates correct behavior
   def test_parse_date_with_invalid_day_raises_error():
       with pytest.raises(ValueError, match="Invalid day: 32"):
           parse_date("2024-01-32")
   ```

2. **Check for over-mocking**:
   ```python
   # Suspicious: Mocking too much
   @patch('module.validate_input', return_value=True)
   @patch('module.process_data', return_value={'status': 'ok'})
   @patch('module.save_result', return_value=None)
   def test_workflow(mock_save, mock_process, mock_validate):
       result = run_workflow(data)
       assert result == {'status': 'ok'}  # Not testing real behavior!

   # Better: Only mock external dependencies
   @patch('module.external_api_call')
   def test_workflow(mock_api):
       mock_api.return_value = expected_api_response
       result = run_workflow(data)
       # Actually tests the real workflow logic
       assert result['processed_count'] == 3
   ```

3. **Identify tests that validate implementation details**:
   ```python
   # Bad: Testing internal implementation
   def test_cache_uses_dictionary():
       cache = Cache()
       assert isinstance(cache._internal_storage, dict)

   # Good: Testing behavior
   def test_cache_retrieves_stored_values():
       cache = Cache()
       cache.set('key', 'value')
       assert cache.get('key') == 'value'
   ```

#### 4.2: Check for Missing Edge Cases

1. **Verify boundary conditions are tested**:
   - Empty inputs
   - None/null values
   - Maximum/minimum values
   - Invalid inputs

2. **Check error handling**:
   - Are error conditions tested?
   - Do tests verify error messages?
   - Are exceptions properly caught?

3. **Look for missing negative tests**:
   ```python
   # If you have:
   def test_valid_input_succeeds(): ...

   # You should also have:
   def test_invalid_input_raises_error(): ...
   ```

#### 4.3: Verify Test Independence

1. **Check for test order dependencies**:
   ```bash
   # Run tests in random order
   uv run pytest tests/path/ --random-order

   # Run single test in isolation
   uv run pytest tests/path/test_file.py::test_name
   ```

2. **Look for shared state issues**:
   - Are tests modifying global state?
   - Do tests depend on previous tests?
   - Are fixtures properly isolated?

#### 4.4: Cross-Reference with Implementation

1. **Read the implementation files**:
   - For each test file, read the corresponding implementation
   - Understand what the code actually does

2. **Compare test expectations to implementation**:
   - Does implementation match test assumptions?
   - Are there code paths not covered by tests?
   - Are there TODOs or FIXMEs that tests don't address?

3. **Look for "convenient" test data**:
   ```python
   # Suspicious: Test uses data that makes bugs invisible
   def test_concatenate_strings():
       result = concatenate("", "")  # Empty strings hide bugs
       assert result == ""

   # Better: Test with realistic data
   def test_concatenate_strings():
       result = concatenate("hello", "world")
       assert result == "hello world"
   ```

**Success Criteria**:
- [ ] Tests validate behavior, not implementation details
- [ ] Tests use realistic, non-trivial test data
- [ ] Mocking is minimal and only for external dependencies
- [ ] Tests are independent and can run in any order
- [ ] Edge cases and error conditions are tested
- [ ] Tests would catch real bugs if implementation broke

### Step 5: Test Quality Metrics

Run automated test quality checks:

1. **Test Coverage**:
   ```bash
   uv run pytest tests/path/ --cov=module --cov-report=term-missing
   ```
   - Check line coverage percentage
   - Identify uncovered critical paths
   - Note: 100% coverage doesn't mean good tests!

2. **Mutation Testing** (if available):
   ```bash
   # mutmut or similar tool
   mutmut run --paths-to-mutate=module/
   ```
   - Checks if tests catch intentional bugs
   - High mutation kill rate = good tests

3. **Test Performance**:
   ```bash
   uv run pytest tests/path/ --durations=10
   ```
   - Identify slow tests
   - Check if tests could be optimized

**Success Criteria**:
- [ ] Coverage meets project standards (>80% for critical paths)
- [ ] No obvious untested code paths
- [ ] Tests run in reasonable time
- [ ] Mutation tests show tests catch bugs (if applicable)

## Validation Report Generation

After completing all analyses, generate a comprehensive report:

```markdown
## Test Verification Report: [Feature Name]

**Plan**: `thoughts/shared/plans/[plan_name]_tdd.md`
**Test Files Verified** (uncommitted only): `tests/path/to/test_*.py`
**Validation Date**: [Date]
**Scope**: Only uncommitted/modified test files

---

### Overall Assessment

✓ **PASS** - Tests are high quality and ready for commit
⚠️ **NEEDS IMPROVEMENT** - Issues identified that should be addressed
❌ **FAIL** - Critical issues must be fixed before commit

---

### 1. Completeness Analysis

**Planned Tests**: 15
**Implemented Tests**: 14
**Extra Tests**: 2

#### Missing Tests:
- ❌ `test_edge_case_with_negative_values` - Planned but not found
  - **Location**: Should be in `tests/path/test_module.py`
  - **Impact**: Medium - Edge case not covered

#### Extra Tests (Not in Plan):
- ✓ `test_performance_with_large_dataset` - Good addition
  - **Location**: `tests/path/test_module.py:234`
  - **Assessment**: Valuable test, recommend adding to plan retrospectively

#### Verdict:
⚠️ **Mostly complete** - One missing test should be added

---

### 2. Atomicity Analysis

**Tests Analyzed**: 16
**Atomic Tests**: 14
**Non-Atomic Tests**: 2

#### Issues Found:

##### Test: `test_user_workflow` (tests/path/test_workflow.py:45)
❌ **Not Atomic** - Tests multiple unrelated behaviors

**Problem**:
```python
def test_user_workflow():
    # Tests authentication, data processing, AND response formatting
    assert authenticate(user) == True
    assert process_data(data) == expected
    assert format_response(result) == formatted
```

**Recommendation**: Split into three tests:
- `test_authentication_succeeds_with_valid_credentials`
- `test_data_processing_returns_expected_format`
- `test_response_formatting_includes_all_fields`

#### Verdict:
⚠️ **Good atomicity** - 2 tests should be split

---

### 3. Informativeness Analysis

**Tests Analyzed**: 16
**Well-Named Tests**: 15
**Tests with Docstrings**: 12
**Tests with Assertion Messages**: 10

#### Issues Found:

##### Test: `test_parse` (tests/path/test_parser.py:23)
⚠️ **Vague name** - Doesn't describe what is being tested

**Current**:
```python
def test_parse():
    result = parse(data)
    assert result == expected
```

**Recommended**:
```python
def test_parse_json_with_nested_objects_returns_dict():
    """
    Test that JSON parser correctly handles nested object structures.

    This ensures deeply nested JSON is properly converted to
    Python dictionaries without data loss.
    """
    json_input = '{"user": {"name": "Alice", "age": 30}}'
    result = parse(json_input)
    assert result == {"user": {"name": "Alice", "age": 30}}, \
        f"Parser returned unexpected structure: {result}"
```

##### Test: `test_division_by_zero` (tests/math/test_calculator.py:67)
⚠️ **Missing assertion message**

**Current**:
```python
assert result is None  # No diagnostic message
```

**Recommended**:
```python
assert result is None, \
    f"Division by zero should return None, got {result}"
```

#### Verdict:
⚠️ **Mostly informative** - 4 tests need better names/messages

---

### 4. Implementation Compensation Analysis

**Tests Analyzed**: 16
**Tests Validating Behavior**: 13
**Tests with Issues**: 3

#### Critical Issues:

##### Test: `test_validate_email` (tests/validators/test_email.py:12)
❌ **CRITICAL: Test is too lenient and hides bugs**

**Problem**:
```python
def test_validate_email():
    result = validate_email("not-an-email")
    assert result is not None  # Just checks it returns something!
```

**What's Wrong**:
- Test passes even when validation incorrectly accepts invalid emails
- Should explicitly test for `False` or exception

**Implementation Review**:
```python
def validate_email(email):
    return email  # BUG: No validation happens!
    # Test passes because "not-an-email" is not None
```

**Fix Required**:
```python
def test_validate_email_rejects_invalid_format():
    """Test that emails without @ symbol are rejected."""
    result = validate_email("not-an-email")
    assert result is False, \
        "Invalid email should be rejected"

def test_validate_email_accepts_valid_format():
    """Test that properly formatted emails are accepted."""
    result = validate_email("user@example.com")
    assert result is True, \
        "Valid email should be accepted"
```

##### Test: `test_data_processing` (tests/path/test_processor.py:45)
⚠️ **Over-mocking hides logic bugs**

**Problem**:
```python
@patch('module.validate')
@patch('module.transform')
@patch('module.save')
def test_data_processing(mock_save, mock_transform, mock_validate):
    # All logic is mocked - not testing anything real!
    mock_validate.return_value = True
    mock_transform.return_value = processed
    mock_save.return_value = None

    result = process_pipeline(data)
    assert result == 'success'
```

**Recommendation**:
- Only mock external I/O (database, API calls)
- Test the actual validation and transformation logic
- Use real test data

##### Test: `test_cache_implementation` (tests/cache/test_cache.py:89)
⚠️ **Testing implementation details**

**Problem**:
```python
def test_cache_uses_lru_strategy():
    cache = Cache()
    # Tests internal _lru_cache attribute
    assert hasattr(cache, '_lru_cache')
```

**Why This Is Bad**:
- Test breaks if implementation changes (e.g., switching to different cache strategy)
- Doesn't verify the actual behavior users care about

**Better Approach**:
```python
def test_cache_evicts_least_recently_used_items():
    """Test that cache removes old items when full."""
    cache = Cache(max_size=2)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.get('a')  # Access 'a' to make it more recent
    cache.set('c', 3)  # Should evict 'b'

    assert cache.get('a') == 1, "Recently accessed item should remain"
    assert cache.get('b') is None, "Least recently used should be evicted"
    assert cache.get('c') == 3, "New item should be cached"
```

#### Missing Edge Cases:

- ❌ No tests for `None` inputs
- ❌ No tests for empty list/dict inputs
- ❌ No tests for maximum integer values
- ⚠️ Error messages not validated (only exception type checked)

#### Test Independence Issues:

**Found**: None - all tests run successfully in random order ✓

#### Verdict:
❌ **CRITICAL ISSUES FOUND** - Must fix test compensation problems

---

### 5. Test Quality Metrics

#### Coverage:
```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
module/core.py            156     12    92%   23-25, 45, 67-70
module/validators.py       45     15    67%   12-26
-----------------------------------------------------
TOTAL                     201     27    87%
```

**Assessment**:
- ✓ Core module has good coverage
- ⚠️ Validators module under-tested (67%)
- Critical: Lines 12-26 in validators.py (email validation) not covered

#### Test Performance:
```
slowest 5 durations:
3.21s test_integration_full_workflow
0.45s test_database_query_performance
0.23s test_large_file_processing
0.12s test_api_call_with_retry
0.08s test_concurrent_requests
```

**Assessment**:
- ⚠️ Integration test is slow (3.2s) - consider optimizing
- ✓ Unit tests are fast

---

## Summary and Recommendations

**Note**: This verification only analyzed uncommitted test files. Already committed tests were not re-verified.

### Critical Issues (Must Fix Before Commit):
1. ❌ **`test_validate_email` hides implementation bug** (tests/validators/test_email.py:12)
   - **Action**: Rewrite test to explicitly check for True/False
   - **Urgency**: HIGH - Current test passes even though validation is broken

### Important Issues (Should Fix):
1. ⚠️ **Missing edge case tests** for None/empty inputs
   - **Action**: Add tests for edge cases
   - **Effort**: 1-2 hours

2. ⚠️ **Over-mocking in `test_data_processing`** (tests/path/test_processor.py:45)
   - **Action**: Reduce mocking to only external dependencies
   - **Effort**: 30 minutes

3. ⚠️ **Low coverage on validators module** (67%)
   - **Action**: Add tests for lines 12-26
   - **Effort**: 1 hour

### Minor Issues (Nice to Have):
1. ⚠️ Improve test naming for 4 tests
2. ⚠️ Add assertion messages to 6 tests
3. ⚠️ Split 2 non-atomic tests

### Strengths:
- ✓ Good test organization and structure
- ✓ Tests are independent (run in any order)
- ✓ Good coverage on core module (92%)
- ✓ Most tests are atomic and well-named

---

## Action Items

Create TodoWrite checklist:
- [ ] Fix critical bug in test_validate_email
- [ ] Add edge case tests for None/empty inputs
- [ ] Reduce mocking in test_data_processing
- [ ] Improve validator test coverage to >80%
- [ ] Improve naming for 4 tests
- [ ] Split 2 non-atomic tests

**Estimated Time to Address**: 3-4 hours

**Recommendation**: ❌ **Do not commit yet** - Fix critical issues first

---

## Detailed Findings

[For each test file, provide detailed analysis...]

### tests/path/test_module.py

**Overall Quality**: Good ✓

**Test List**:
1. ✓ `test_basic_functionality` - Atomic, informative, validates behavior
2. ✓ `test_edge_case_empty_input` - Atomic, informative, validates behavior
3. ⚠️ `test_parse` - Vague name, needs improvement
...

[Continue for each test file...]

```

## Important Guidelines

1. **Git-First Approach**:
   - ALWAYS start by checking `git status --porcelain tests/`
   - Only verify tests that are modified (M) or untracked (??)
   - If no uncommitted test files, inform user and exit gracefully
   - This prevents re-verifying already reviewed and committed tests

2. **Be Thorough but Constructive**:
   - Point out issues clearly
   - Explain *why* something is a problem
   - Provide concrete examples of how to fix
   - Acknowledge good testing practices

3. **Focus on Real Issues**:
   - Don't nitpick style if tests are functionally good
   - Prioritize tests that hide bugs over naming issues
   - Focus on test behavior, not test implementation

4. **Provide Context**:
   - Show code snippets
   - Include file:line references
   - Explain the impact of issues
   - Differentiate critical vs minor issues

5. **Be Skeptical**:
   - Question if tests really validate what they claim
   - Look for tests that pass for wrong reasons
   - Check if test data is realistic
   - Verify tests would catch real bugs

6. **Use Automation**:
   - Run tests multiple times
   - Try random order execution
   - Check coverage reports
   - Use mutation testing if available

## Verification Checklist

For each test in the plan:
- [ ] Test exists in codebase
- [ ] Test is atomic (tests one thing)
- [ ] Test name is descriptive
- [ ] Test has informative docstring
- [ ] Test has diagnostic assertion messages
- [ ] Test validates behavior, not implementation
- [ ] Test uses realistic data
- [ ] Test doesn't over-mock
- [ ] Test is independent
- [ ] Test would catch bugs if implementation broke

## Common Test Smells to Detect

1. **Too Lenient**:
   - `assert result is not None` (instead of checking actual value)
   - `assert len(result) > 0` (instead of checking contents)
   - Only testing happy path

2. **Over-Mocking**:
   - Mocking internal functions
   - Mocking everything, testing nothing
   - Mock return values match expected values exactly

3. **Testing Implementation**:
   - Checking internal state/attributes
   - Verifying algorithm steps
   - Testing private methods directly

4. **Not Atomic**:
   - Test name includes "and"
   - Multiple unrelated assertions
   - Would need multiple fixes if it failed

5. **Not Independent**:
   - Tests fail when run in isolation
   - Tests modify global state
   - Tests depend on execution order

6. **Poor Diagnostics**:
   - Vague test names
   - No docstrings
   - No assertion messages
   - Unclear failure output

## Usage Example

```bash
# After implementing a TDD plan (will only verify uncommitted test files)
/verify_tests thoughts/shared/plans/onnx-conv2d-tdd.md

# Or let it discover the plan (will only verify uncommitted test files)
/verify_tests

# Note: The command automatically checks git status and only verifies
# test files that are modified (M) or untracked (??).
# If no uncommitted test files exist, it will inform you and exit.
```

## Integration with Other Commands

Recommended workflow:
1. `/create_plan_tdd` - Create TDD implementation plan
2. `/implement_plan` - Implement following TDD approach
3. `/verify_tests` - Verify test quality (this command)
4. `/validate_plan` - Verify overall implementation
5. `/commit` - Commit changes
6. `/describe_pr` - Generate PR description

This command focuses specifically on test quality, while `/validate_plan` focuses on overall implementation correctness.

## Why Git-First?

This command only verifies uncommitted test files because:
- **Efficiency**: Avoids re-analyzing already reviewed and committed tests
- **Focus**: Concentrates on the tests you're actively working on
- **Workflow Integration**: Fits naturally into the TDD cycle (write test → verify → commit)
- **Incremental Validation**: Ensures each batch of tests is validated before commit

If you need to verify all tests (including committed ones), you can temporarily unstage or modify them, or create a separate validation command for comprehensive test suite audits.

Remember: The goal is to ensure tests are trustworthy guardians of code quality, not just checkboxes for coverage metrics.
