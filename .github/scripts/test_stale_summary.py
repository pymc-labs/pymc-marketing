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
"""Offline tests for the model-free stale-issue summary."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "stale", Path(__file__).with_name("stale_summary.py")
)
stale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stale)

NOW = datetime(2026, 10, 23, tzinfo=UTC)


def event(typename, created, **fields):
    """Build a GraphQL-shaped timeline event."""
    return {"__typename": typename, "createdAt": created, **fields}


def issue(number, **fields):
    """Build a GraphQL-shaped open issue."""
    return {
        "number": number,
        "body": "",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2026-10-01T00:00:00Z",
        "labels": {"nodes": []},
        "assignees": {"totalCount": 0},
        "comments": {"nodes": []},
        "timelineItems": {"nodes": []},
    } | fields


def merged_pr(number, body="", base="main", closing=(), merged="2026-07-01T00:00:00Z"):
    """Build a GraphQL-shaped merged PR."""
    return {
        "number": number,
        "title": "",
        "body": body,
        "baseRefName": base,
        "mergedAt": merged,
        "closingIssuesReferences": {"nodes": [{"number": n} for n in closing]},
    }


def comment(body, login="someone", association="NONE", created="2026-08-01T00:00:00Z"):
    """Build a GraphQL-shaped issue comment."""
    return {
        "author": {"login": login},
        "authorAssociation": association,
        "createdAt": created,
        "body": body,
    }


def result(**sections):
    """Build a valid artifact."""
    data = {"date": "2026-10-23", "commit": "3a0dbba4"}
    return data | {key: sections.get(key, []) for key in stale.SECTIONS}


class StaleSummaryTests(unittest.TestCase):
    """Exercise the candidate rules, the artifact schema and publication."""

    def setUp(self):
        """Isolate each test from real credentials and filesystem outputs."""
        env = patch.dict(
            os.environ, {"GITHUB_REPOSITORY": "test/repo", "GH_TOKEN": "test-token"}
        )
        env.start()
        self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        old = os.getcwd()
        os.chdir(temp.name)
        self.addCleanup(os.chdir, old)

    def candidates(self, merged=(), open_prs=(), issues=(), index=(set(), set())):
        """Run the candidate rules against fixture data."""
        return stale.candidates(
            list(merged), "main", list(open_prs), list(issues), index, NOW
        )

    def test_merged_pr_reasons(self):
        """Report merges into other branches, unticked checklist keywords and misses."""
        reopened = {"nodes": [event("ReopenedEvent", "2026-08-01T00:00:00Z")]}
        found = self.candidates(
            merged=[
                merged_pr(1, "Fixes #10", base="v1.0.0", closing=[10]),
                merged_pr(2, "- [ ] Closes https://github.com/test/repo/issues/11"),
                merged_pr(3, "Resolves #12"),
                merged_pr(4, "Fixes #13"),
            ],
            issues=[issue(10), issue(11), issue(12), issue(13, timelineItems=reopened)],
        )
        self.assertEqual(
            found["closed_by_merge"],
            [
                {"issue": 10, "pr": 1, "reason": "base", "base": "v1.0.0"},
                {"issue": 11, "pr": 2, "reason": "checklist"},
                {"issue": 12, "pr": 3, "reason": "keyword"},
            ],
        )

    def test_other_repository_references_ignored(self):
        """Ignore closing references that point at another repository."""
        found = self.candidates(
            merged=[
                merged_pr(1, "Fixes pymc-devs/pymc#10"),
                merged_pr(2, "Closes https://github.com/other/repo/issues/11"),
                merged_pr(3, "Fixes https://github.com/Test/Repo/issues/12"),
            ],
            issues=[issue(10), issue(11), issue(12)],
        )
        self.assertEqual([e["issue"] for e in found["closed_by_merge"]], [12])

    def test_weak_references_only_from_recent_merges(self):
        """List Refs mentions from the last 35 days, skipping issues already listed."""
        found = self.candidates(
            merged=[
                merged_pr(1, "Refs #10", merged="2026-10-10T00:00:00Z"),
                merged_pr(2, "Refs #11", merged="2026-06-01T00:00:00Z"),
                merged_pr(
                    3, "Fixes #12\nRelated to #12", merged="2026-10-10T00:00:00Z"
                ),
            ],
            issues=[issue(10), issue(11), issue(12)],
        )
        self.assertEqual(found["referenced"], [{"issue": 10, "pr": 1}])

    def test_idle_assignees_claims_and_prs(self):
        """List quiet assignees, unanswered claims and idle linked PRs."""
        quiet = "2026-01-01T00:00:00Z"
        later_pr = event(
            "CrossReferencedEvent",
            "2026-09-01T00:00:00Z",
            source={"author": {"login": "c"}},
        )
        found = self.candidates(
            open_prs=[
                {
                    "number": 50,
                    "body": "Closes #3",
                    "updatedAt": "2026-05-01T00:00:00Z",
                    "closingIssuesReferences": {"nodes": []},
                }
            ],
            issues=[
                issue(
                    1,
                    updatedAt=quiet,
                    assignees={"totalCount": 1},
                    timelineItems={
                        "nodes": [event("AssignedEvent", "2025-03-01T00:00:00Z")]
                    },
                ),
                issue(2, comments={"nodes": [comment("I'd like to work on this!")]}),
                issue(3, updatedAt=quiet, assignees={"totalCount": 1}),
                issue(
                    4,
                    comments={
                        "nodes": [comment("Can I take this?", association="MEMBER")]
                    },
                ),
                issue(
                    5,
                    comments={"nodes": [comment("May I be assigned?", login="c")]},
                    timelineItems={"nodes": [later_pr]},
                ),
                issue(
                    6,
                    comments={
                        "nodes": [
                            comment("I'll work on it", created="2026-10-20T00:00:00Z")
                        ]
                    },
                ),
            ],
        )
        self.assertEqual(found["idle_assignees"], [{"issue": 1, "since": "2025-03-01"}])
        self.assertEqual(found["idle_claims"], [{"issue": 2, "since": "2026-08-01"}])
        self.assertEqual(
            found["idle_prs"], [{"issue": 3, "pr": 50, "updated": "2026-05-01"}]
        )

    def test_removed_code_references(self):
        """Flag bug reports and path-citing issues whose references are all gone."""
        bug = {"nodes": [{"name": "bug"}]}
        found = self.candidates(
            index=({"pymc_marketing/mmm/mmm.py"}, {"sample_curve"}),
            issues=[
                issue(1, labels=bug, body="`old_method()` fails in `OldClass`"),
                issue(2, body="Add `new_method` and `NewThing`"),
                issue(3, body="pymc_marketing/mmm/base.py breaks `legacy_fn`"),
                issue(4, labels=bug, body="`sample_curve` and `missing_fn` disagree"),
            ],
        )
        self.assertEqual(
            found["removed_code"],
            [
                {"issue": 1, "tokens": ["OldClass", "old_method"]},
                {"issue": 3, "tokens": ["pymc_marketing/mmm/base.py", "legacy_fn"]},
            ],
        )

    def test_injected_values_rejected(self):
        """Reject anything but numbers, dates, branch names and plain code tokens."""
        bad = [
            {"closed_by_merge": [{"issue": "1", "pr": 2, "reason": "keyword"}]},
            {"closed_by_merge": [{"issue": 1, "pr": 2, "reason": "@everyone"}]},
            {"closed_by_merge": [{"issue": 1, "pr": 2, "reason": ["base"]}]},
            {
                "closed_by_merge": [
                    {"issue": 1, "pr": 2, "reason": "base", "base": "x`; @user"}
                ]
            },
            {"idle_claims": [{"issue": 1, "since": "@user"}]},
            {"idle_claims": [{"issue": 1, "since": "2026-01-01", "login": "x"}]},
            {"removed_code": [{"issue": 1, "tokens": ["@user"]}]},
            {"removed_code": [{"issue": 1, "tokens": ["a`b"]}]},
            {"removed_code": [{"issue": 1, "tokens": []}]},
            {"referenced": [{"issue": True, "pr": 2}]},
            {"referenced": [[1, 2]]},
            {"idle_prs": ["#1"]},
        ]
        for sections in bad:
            with self.subTest(sections=sections), self.assertRaises(ValueError):
                stale.validate(result(**sections))
        for extra in [{"title": "x"}, {"commit": "$(id)"}, {"date": "today"}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                stale.validate(result() | extra)

    def test_render_uses_fixed_text_and_numbers(self):
        """Render only marker, fixed headings and validated values."""
        body = stale.render(
            result(
                closed_by_merge=[
                    {"issue": 10, "pr": 1, "reason": "base", "base": "v1.0.0"}
                ],
                removed_code=[{"issue": 3, "tokens": ["legacy_fn"]}],
            )
        )
        self.assertTrue(body.startswith(stale.MARKER))
        self.assertIn("- #10 via #1: merged into `v1.0.0`", body)
        self.assertIn("- #3: `legacy_fn`", body)
        self.assertEqual(body.count("None."), 4)

    def test_publish_revalidates_and_limits_size(self):
        """Fail before any API call on invalid or oversized artifacts."""
        for content in [
            json.dumps(result(idle_claims=[{"issue": 1, "since": "x"}])),
            " " * 60001,
        ]:
            Path("result.json").write_text(content)
            with (
                self.subTest(size=len(content)),
                patch.object(stale, "github") as github,
                self.assertRaises(ValueError),
            ):
                stale.publish()
            github.assert_not_called()

    def test_publish_updates_only_this_month_bot_issue(self):
        """Patch this month's bot issue; ignore spoofs and create one otherwise."""
        Path("result.json").write_text(json.dumps(result()))
        title = "Stale-issue candidates 2026-10"
        own = {
            "number": 7,
            "title": title,
            "body": stale.MARKER,
            "user": {"login": "github-actions[bot]"},
        }
        spoof = own | {"number": 8, "user": {"login": "attacker"}}
        old = own | {"number": 9, "title": "Stale-issue candidates 2026-09"}
        with patch.object(stale, "github", side_effect=[[spoof, old, own], {}]) as gh:
            stale.publish()
        self.assertEqual(gh.call_args.args[:2], ("/repos/test/repo/issues/7", "PATCH"))
        with patch.object(stale, "github", side_effect=[[spoof, old], {}]) as gh:
            stale.publish()
        path, method, payload = gh.call_args.args
        self.assertEqual((path, method), ("/repos/test/repo/issues", "POST"))
        self.assertEqual(
            (payload["title"], payload["labels"]), (title, ["maintenance"])
        )

    def test_graphql_pages_through_github_helper(self):
        """Page with the job token helper and fail on GraphQL errors."""
        page = {
            "data": {
                "repository": {
                    "issues": {
                        "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                        "nodes": [1],
                    }
                }
            }
        }
        last = json.loads(json.dumps(page))
        last["data"]["repository"]["issues"] = {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [2],
        }
        with patch.object(stale, "github", side_effect=[page, last]) as gh:
            self.assertEqual(stale.graphql_nodes("query", "issues"), [1, 2])
        self.assertEqual(gh.call_args.args[:2], ("/graphql", "POST"))
        self.assertEqual(gh.call_args.args[2]["variables"]["cursor"], "c1")
        with (
            patch.object(stale, "github", return_value={"errors": [{}]}),
            self.assertRaises(ValueError),
        ):
            stale.graphql_nodes("query", "issues")


if __name__ == "__main__":
    unittest.main()
