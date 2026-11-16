#!/usr/bin/env python3
"""Closing-issue and assignee checks for the first-pull-request workflow.

GitHub closing keywords count in the pull request body and in commit messages.
They do not count in the title. A bare ``#123`` is not a closing reference.
A reference to another repository is not a closing reference for this one.

The workflow passes title, body, and commit messages as files. This module
does not read those texts from the shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

POLICY_SECTION_URL = (
    "https://github.com/pymc-labs/pymc-marketing/blob/main/"
    "CONTRIBUTING.md#first-pull-requests"
)
POLICY_COMMENT_MARKER = "<!-- pymc-marketing-first-pr-assignment -->"
LABEL_NAME = "spam?"
LABEL_COLOR = "FBCA04"
LABEL_DESCRIPTION = (
    "First pull request does not fix an issue the author is assigned to. "
    "Triage flag, not a rejection."
)
# A member token sees FIRST_TIMER or FIRST_TIME_CONTRIBUTOR for a new author.
# The workflow token cannot: pulls.get with that token, like an unauthenticated
# read, returns NONE for those authors and still returns MEMBER, OWNER,
# COLLABORATOR, or CONTRIBUTOR for everyone else. Flag the values this job
# can actually observe. Do not flag the established roles.
FIRST_TIME_ASSOCIATIONS = frozenset({"FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR", "NONE"})
# Same substring checks as the job condition in first-pr-assignment.yml.
# GitHub's contains() is case-sensitive, so these markers are too.
BOT_ACTOR_MARKERS = ("[bot]", "dependabot", "renovate")

# Keyword, optional colon, then one reference. GitHub's documented forms are
# ``#N``, ``owner/repo#N``, and a github.com issues URL. One keyword closes
# one reference: ``Fixes #1, #2`` does not close 2.
_NAME = r"[A-Za-z0-9_.-]+"
_CLOSING_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
    r"(?:[ \t]*:[ \t]*|[ \t]+)"
    r"(?:"
    r"https?://(?:www\.)?github\.com/"
    rf"(?P<url_owner>{_NAME})/(?P<url_repo>{_NAME})/issues/(?P<url_num>\d+)(?!/)"
    r"|"
    rf"(?P<qual_owner>{_NAME})/(?P<qual_repo>{_NAME})#(?P<qual_num>\d+)"
    r"|"
    r"#(?P<num>\d+)"
    r")"
    r"\b",
    re.IGNORECASE,
)


def closing_issue_numbers(text: str, *, owner: str, repo: str) -> frozenset[int]:
    """Return issue numbers in ``owner/repo`` that ``text`` would close."""
    expected_owner = owner.casefold()
    expected_repo = repo.casefold()
    found: set[int] = set()
    for match in _CLOSING_REFERENCE.finditer(text):
        number = _number_for_this_repo(match, expected_owner, expected_repo)
        if number is not None:
            found.add(number)
    return frozenset(found)


def closing_issue_numbers_from_pull_request(
    *,
    body: str,
    title: str,
    commit_messages: Iterable[str],
    owner: str,
    repo: str,
) -> frozenset[int]:
    """Return issue numbers this pull request would close in ``owner/repo``.

    ``title`` is validated and then ignored. A closing keyword in the title,
    including ``Fixes #123``, does not close an issue.
    """
    if not isinstance(body, str) or not isinstance(title, str):
        raise TypeError("body and title must be strings")
    found: set[int] = set()
    for text in (body, *commit_messages):
        found.update(closing_issue_numbers(text, owner=owner, repo=repo))
    return frozenset(found)


def author_is_assignee(author: str, assignees: Iterable[str]) -> bool:
    """Return whether ``author`` matches an assignee login, ignoring case."""
    if not author:
        return False
    folded = author.casefold()
    return any(assignee.casefold() == folded for assignee in assignees if assignee)


def should_flag(
    *,
    actor: str,
    author_association: str,
    author: str,
    issues: Iterable[Mapping[str, object]],
) -> bool:
    """Return whether this pull request gets the triage label and comment.

    ``author_association`` is the pull request API value read by the workflow
    token, not a member's view of the same field and not the
    ``pull_request_target`` event value. That token reports ``NONE`` for
    first-time authors. ``issues`` are closing references loaded from this
    repository. A missing issue is omitted by the caller. A pull request
    payload does not count as an issue assignment. Pass if any remaining
    issue lists ``author``.
    """
    if is_skipped_bot(actor):
        return False
    if author_association not in FIRST_TIME_ASSOCIATIONS:
        return False
    return not any(_assigned_issue(author, issue) for issue in issues)


def is_skipped_bot(actor: str) -> bool:
    """Return whether ``actor`` matches the workflow bot skip."""
    return any(marker in actor for marker in BOT_ACTOR_MARKERS)


def label_spec() -> dict[str, str]:
    """Return the triage label name, color, and description."""
    return {
        "name": LABEL_NAME,
        "color": LABEL_COLOR,
        "description": LABEL_DESCRIPTION,
    }


def policy_comment() -> str:
    """Return the one comment posted when a first pull request is flagged."""
    paragraphs = [
        "Thank you for opening a pull request.",
        (
            "Our policy for first-time contributors is that a maintainer assigns "
            "you to an issue before we accept a pull request. We do this because "
            "a flood of unsolicited pull requests was taking review time away "
            "from work we had asked for. The policy is in "
            f"[CONTRIBUTING.md]({POLICY_SECTION_URL})."
        ),
        (
            "This pull request does not fix an issue you are assigned to, so it "
            "has been labeled `spam?`. That label is a question for maintainers, "
            "not a claim that you or this change are spam. A useful fix can "
            "arrive without that assignment."
        ),
        (
            "If this is one of those, reply here and a maintainer will look. If "
            "you are already assigned, reply with the issue number and use a "
            "closing keyword such as `Fixes #123` in the pull request "
            "description. Otherwise, please close this pull request, comment on "
            "the issue you want to work on, and wait for an assignment. We will "
            "not close it for you."
        ),
        POLICY_COMMENT_MARKER,
    ]
    return "\n\n".join(paragraphs) + "\n"


def commit_messages_from_api_payload(payload: object) -> list[str]:
    """Return commit messages from a pulls.listCommits JSON payload."""
    if not isinstance(payload, list):
        raise ValueError("pull request commits payload must be a JSON list")
    messages: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("pull request commit entry must be an object")
        commit = item.get("commit")
        message = commit.get("message") if isinstance(commit, dict) else None
        if not isinstance(message, str):
            raise ValueError("pull request commit entry is missing commit.message")
        messages.append(message)
    return messages


def _number_for_this_repo(
    match: re.Match[str], expected_owner: str, expected_repo: str
) -> int | None:
    url_number = match.group("url_num")
    if url_number is not None:
        if not _same_repo(
            match.group("url_owner"),
            match.group("url_repo"),
            expected_owner,
            expected_repo,
        ):
            return None
        return int(url_number)
    qualified_number = match.group("qual_num")
    if qualified_number is not None:
        if not _same_repo(
            match.group("qual_owner"),
            match.group("qual_repo"),
            expected_owner,
            expected_repo,
        ):
            return None
        return int(qualified_number)
    return int(match.group("num"))


def _same_repo(owner: str, repo: str, expected_owner: str, expected_repo: str) -> bool:
    return owner.casefold() == expected_owner and repo.casefold() == expected_repo


def _assigned_issue(author: str, issue: Mapping[str, object]) -> bool:
    if _is_pull_request(issue):
        return False
    return author_is_assignee(author, _assignee_logins(issue))


def _is_pull_request(issue: Mapping[str, object]) -> bool:
    return "pull_request" in issue and issue.get("pull_request") is not None


def _assignee_logins(issue: Mapping[str, object]) -> list[str]:
    raw = issue.get("assignees") or []
    if not isinstance(raw, list):
        raise ValueError("issue assignees must be a list")
    logins: list[str] = []
    for assignee in raw:
        if isinstance(assignee, str):
            logins.append(assignee)
        elif isinstance(assignee, dict):
            login = assignee.get("login")
            if not isinstance(login, str):
                raise ValueError("issue assignee is missing login")
            logins.append(login)
        else:
            raise ValueError("issue assignee must be a login or object")
    return logins


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _command_closing_issues(args: argparse.Namespace) -> int:
    numbers = closing_issue_numbers_from_pull_request(
        body=_read_text(args.body_file),
        title=_read_text(args.title_file),
        commit_messages=commit_messages_from_api_payload(_read_json(args.commits_file)),
        owner=args.owner,
        repo=args.repo,
    )
    json.dump(sorted(numbers), sys.stdout)
    sys.stdout.write("\n")
    return 0


def _command_should_flag(args: argparse.Namespace) -> int:
    payload = _read_json(args.issues_file)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("issues payload must be a JSON list of objects")
    flagged = should_flag(
        actor=args.actor,
        author_association=args.association,
        author=args.author,
        issues=payload,
    )
    json.dump({"flag": flagged}, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _command_label_spec(_args: argparse.Namespace) -> int:
    json.dump(label_spec(), sys.stdout)
    sys.stdout.write("\n")
    return 0


def _command_policy_comment(_args: argparse.Namespace) -> int:
    sys.stdout.write(policy_comment())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    closing = commands.add_parser("closing-issues")
    closing.add_argument("--body-file", required=True)
    closing.add_argument("--title-file", required=True)
    closing.add_argument("--commits-file", required=True)
    closing.add_argument("--owner", required=True)
    closing.add_argument("--repo", required=True)
    closing.set_defaults(func=_command_closing_issues)

    decision = commands.add_parser("should-flag")
    decision.add_argument("--issues-file", required=True)
    decision.add_argument("--author", default="")
    decision.add_argument("--actor", default="")
    decision.add_argument("--association", default="")
    decision.set_defaults(func=_command_should_flag)

    label = commands.add_parser("label-spec")
    label.set_defaults(func=_command_label_spec)

    comment = commands.add_parser("policy-comment")
    comment.set_defaults(func=_command_policy_comment)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a subcommand. User-controlled text is read from files, not argv."""
    args = _parser().parse_args(argv)
    if args.command == "should-flag":
        args.author = args.author or _env("PR_AUTHOR")
        args.actor = args.actor or _env("PR_ACTOR")
        args.association = args.association or _env("PR_ASSOCIATION")
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"first_pr_assignment: {exc}", file=sys.stderr)
        return 1


def _env(name: str) -> str:
    return os.environ.get(name, "")


if __name__ == "__main__":
    raise SystemExit(main())
