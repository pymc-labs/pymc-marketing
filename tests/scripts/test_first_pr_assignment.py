#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for the first-pull-request assignment check."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "first_pr_assignment.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"
OWNER = "pymc-labs"
REPO = "pymc-marketing"


def _load_script_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("first_pr_assignment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def _numbers(script_module, text: str) -> frozenset[int]:
    return script_module.closing_issue_numbers(text, owner=OWNER, repo=REPO)


def _issue(*logins: str, pull_request: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "assignees": [{"login": login} for login in logins],
    }
    if pull_request:
        payload["pull_request"] = {"url": "https://api.github.com/pulls/1"}
    return payload


def test_closing_keyword_in_body(script_module) -> None:
    assert _numbers(script_module, "Fixes #1428") == frozenset({1428})


@pytest.mark.parametrize(
    "keyword",
    [
        "close",
        "closes",
        "closed",
        "fix",
        "fixes",
        "fixed",
        "resolve",
        "resolves",
        "resolved",
    ],
)
def test_each_documented_closing_keyword(script_module, keyword: str) -> None:
    assert _numbers(script_module, f"{keyword.upper()}: #15") == frozenset({15})


def test_bare_issue_number_does_not_count(script_module) -> None:
    assert _numbers(script_module, "See #1428 and GH-1428") == frozenset()


def test_title_only_closing_keyword_does_not_count(script_module) -> None:
    found = script_module.closing_issue_numbers_from_pull_request(
        body="See #9",
        title="Fixes #9",
        commit_messages=[],
        owner=OWNER,
        repo=REPO,
    )
    assert found == frozenset()


def test_other_repository_issue_url_does_not_count(script_module) -> None:
    text = "Fixes https://github.com/other/repo/issues/123"
    assert _numbers(script_module, text) == frozenset()


def test_same_repository_issue_url_counts_case_insensitively(script_module) -> None:
    text = "Fixes https://GitHub.com/PyMC-Labs/PyMC-Marketing/issues/44"
    assert _numbers(script_module, text) == frozenset({44})
    extra = "Fixes https://github.com/pymc-labs/pymc-marketing/issues/44/files"
    assert _numbers(script_module, extra) == frozenset()


def test_qualified_reference_is_repository_specific(script_module) -> None:
    assert _numbers(script_module, "Fixes pymc-labs/pymc-marketing#7") == frozenset({7})
    assert _numbers(script_module, "Fixes other/repo#7") == frozenset()
    assert (
        _numbers(script_module, "Fixes pymc-labs/pymc-marketing-fork#7") == frozenset()
    )


def test_commit_message_closing_keyword_counts(script_module) -> None:
    found = script_module.closing_issue_numbers_from_pull_request(
        body="",
        title="Fixes #3",
        commit_messages=["subject only\n\nResolve #3"],
        owner=OWNER,
        repo=REPO,
    )
    assert found == frozenset({3})


def test_keyword_does_not_apply_to_a_following_bare_number(script_module) -> None:
    assert _numbers(script_module, "Fixes #1, #2") == frozenset({1})
    assert _numbers(script_module, "Fixes #1, fixes #2") == frozenset({1, 2})


def test_keyword_must_be_a_whole_word(script_module) -> None:
    text = "prefix #1\nenclose #2\nFixing #3\nDuplicate of #4\nFixes the bug in #5"
    assert _numbers(script_module, text) == frozenset()


def test_newline_and_pull_url_do_not_count(script_module) -> None:
    assert _numbers(script_module, "Fixes\n#8") == frozenset()
    text = "Fixes https://github.com/pymc-labs/pymc-marketing/pull/8"
    assert _numbers(script_module, text) == frozenset()


def test_assignee_login_differs_only_by_case(script_module) -> None:
    assert script_module.author_is_assignee("drbenvincent", ["DrBenVincent"])
    assert not script_module.author_is_assignee("someone-else", ["DrBenVincent"])
    assert not script_module.author_is_assignee("", [""])


def test_assigned_author_is_not_flagged(script_module) -> None:
    flagged = script_module.should_flag(
        actor="drbenvincent",
        author_association="FIRST_TIMER",
        author="drbenvincent",
        issues=[_issue("DrBenVincent")],
    )
    assert flagged is False


def test_unassigned_and_other_assignee_are_flagged(script_module) -> None:
    common = {
        "actor": "new-author",
        "author_association": "FIRST_TIME_CONTRIBUTOR",
        "author": "new-author",
    }
    assert script_module.should_flag(**common, issues=[]) is True
    assert script_module.should_flag(**common, issues=[_issue()]) is True
    assert script_module.should_flag(**common, issues=[_issue("maintainer")]) is True


def test_any_assigned_closing_issue_passes(script_module) -> None:
    flagged = script_module.should_flag(
        actor="new-author",
        author_association="FIRST_TIMER",
        author="new-author",
        issues=[_issue("maintainer"), _issue("New-Author")],
    )
    assert flagged is False


@pytest.mark.parametrize(
    "association",
    ["MEMBER", "OWNER", "COLLABORATOR", "CONTRIBUTOR", "MANNEQUIN", ""],
)
def test_established_associations_are_not_flagged(
    script_module, association: str
) -> None:
    flagged = script_module.should_flag(
        actor="maintainer",
        author_association=association,
        author="maintainer",
        issues=[],
    )
    assert flagged is False


def test_workflow_token_none_is_flagged_unless_assigned(script_module) -> None:
    """The workflow token sees NONE where a member token sees FIRST_TIME_*."""
    assert (
        script_module.should_flag(
            actor="new-author",
            author_association="NONE",
            author="new-author",
            issues=[],
        )
        is True
    )
    assert (
        script_module.should_flag(
            actor="new-author",
            author_association="NONE",
            author="new-author",
            issues=[_issue("new-author")],
        )
        is False
    )


@pytest.mark.parametrize(
    "actor",
    ["dependabot[bot]", "pre-commit-ci[bot]", "renovate", "dependabot"],
)
def test_bot_actors_are_not_flagged(script_module, actor: str) -> None:
    flagged = script_module.should_flag(
        actor=actor,
        author_association="FIRST_TIMER",
        author=actor,
        issues=[],
    )
    assert flagged is False


def test_pull_request_assignment_does_not_count(script_module) -> None:
    flagged = script_module.should_flag(
        actor="new-author",
        author_association="FIRST_TIMER",
        author="new-author",
        issues=[_issue("new-author", pull_request=True)],
    )
    assert flagged is True


def test_cli_ignores_title_and_reads_commit_message(
    script_module, tmp_path: Path
) -> None:
    del script_module
    body = tmp_path / "body.txt"
    title = tmp_path / "title.txt"
    commits = tmp_path / "commits.json"
    body.write_text("See #2\n", encoding="utf-8")
    title.write_text("Fixes #1\n", encoding="utf-8")
    commits.write_text(
        json.dumps([{"commit": {"message": "unrelated\n\nFixes #7"}}]),
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "closing-issues",
            "--body-file",
            str(body),
            "--title-file",
            str(title),
            "--commits-file",
            str(commits),
            "--owner",
            OWNER,
            "--repo",
            REPO,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [7]


def test_cli_should_flag_reads_author_from_environment(
    script_module, tmp_path: Path
) -> None:
    del script_module
    issues = tmp_path / "issues.json"
    issues.write_text(
        json.dumps([{"assignees": [{"login": "Ada"}]}]),
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "should-flag",
            "--issues-file",
            str(issues),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PR_AUTHOR": "ada",
            "PR_ACTOR": "ada",
            "PR_ASSOCIATION": "FIRST_TIMER",
        },
    )
    assert json.loads(result.stdout) == {"flag": False}


def test_cli_rejects_malformed_commits(tmp_path: Path) -> None:
    body = tmp_path / "body.txt"
    title = tmp_path / "title.txt"
    commits = tmp_path / "commits.json"
    body.write_text("", encoding="utf-8")
    title.write_text("", encoding="utf-8")
    commits.write_text("{", encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "closing-issues",
            "--body-file",
            str(body),
            "--title-file",
            str(title),
            "--commits-file",
            str(commits),
            "--owner",
            OWNER,
            "--repo",
            REPO,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_policy_comment_links_to_contributing_section(script_module) -> None:
    comment = script_module.policy_comment()
    assert script_module.POLICY_SECTION_URL in comment
    assert script_module.POLICY_COMMENT_MARKER in comment
    assert "We will not close it for you." in comment

    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    heading = "### First pull requests"
    assert heading in contributing
    section = contributing.split(heading, 1)[1].split("\n## ", 1)[0]
    assert "assignee" in section
    assert "`spam?`" in section
    assert "FIRST_TIMER" in section
    assert "strongly encourage" not in contributing


def test_label_spec_matches_policy(script_module) -> None:
    spec = script_module.label_spec()
    assert spec == {
        "name": "spam?",
        "color": "FBCA04",
        "description": (
            "First pull request does not fix an issue the author is assigned to. "
            "Triage flag, not a rejection."
        ),
    }
    assert len(spec["description"]) <= 100


def test_workflow_reads_association_from_the_pull_request_api() -> None:
    """The event field is NONE for first-time authors, so it cannot be the gate."""
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "first-pr-assignment.yml"
    ).read_text(encoding="utf-8")
    job_if = workflow.split("if: >-", 1)[1].split("permissions:", 1)[0]
    script = workflow.split("script: |", 1)[1]
    assert "author_association" not in job_if
    assert "FIRST_TIMER" not in job_if
    assert "github.rest.pulls.get" in script
    assert "pull.data.author_association" in script
    assert "pullRequest.author_association" not in script
    assert "PR_ASSOCIATION: association" in script
