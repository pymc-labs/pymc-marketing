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
"""List stale-issue candidates without a model call.

The analysis job reads GitHub data with a read-only token and checks code names
cited in issues against the checked-out package. It writes only numbers, dates,
branch names and checked code tokens. A separate job revalidates the artifact,
renders fixed text and creates or updates one bot-authored issue per month.
"""

import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from issue_automation import github

MARKER = "<!-- stale-summary -->"
REFS_DAYS, ASSIGNEE_DAYS, CLAIM_DAYS, PR_DAYS = 35, 180, 42, 90
REF = r"(?:(?<![\w/-])#(\d+)|https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+))"
KEYWORD_RE = re.compile(
    rf"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b:?\s+{REF}", re.I
)
WEAK_RE = re.compile(
    rf"\b(?:refs?|related to|part of|addresses|towards)\b:?\s+{REF}", re.I
)
CLAIM_RE = re.compile(
    r"\b(?:i(?:'d| would) like to (?:work on|take|tackle)|can i (?:work on|take)"
    r"|may i (?:be assigned|work on|take)|assign (?:it |this )?to me"
    r"|i(?:'ll| will) (?:work on|take) (?:this|it)|i can take (?:this|it))\b",
    re.I,
)
CITED_NAME = re.compile(r"`([A-Za-z_][\w.]*?)(?:\(\))?`")
CITED_PATH = re.compile(r"\b(pymc_marketing/[\w/]+\.py)\b")
UNTICKED = re.compile(r"\s*[-*]\s*\[ \]")
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}
REASONS = {
    "base": "merged into `{base}`",
    "checklist": "the closing keyword is in an unticked checklist item",
    "keyword": "the closing keyword did not close it",
}


def _number(value):
    return type(value) is int and 0 < value < 2**31


def _matches(pattern):
    compiled = re.compile(pattern)
    return lambda value: type(value) is str and compiled.fullmatch(value) is not None


_date = _matches(r"\d{4}-\d{2}-\d{2}")
_token = _matches(r"[A-Za-z_][\w./]{0,99}")
CHECKS = {
    "issue": _number,
    "pr": _number,
    "reason": lambda value: type(value) is str and value in REASONS,
    "base": _matches(r"[\w./-]{1,100}"),
    "since": _date,
    "updated": _date,
    "tokens": lambda value: (
        type(value) is list and 0 < len(value) <= 10 and all(map(_token, value))
    ),
}
SECTIONS = {
    "closed_by_merge": (
        {"issue", "pr", "reason"},
        "Merged PR, issue still open",
        lambda e: f"#{e['issue']} via #{e['pr']}: " + REASONS[e["reason"]].format(**e),
    ),
    "referenced": (
        {"issue", "pr"},
        f'Mentioned with "Refs" or "Related to" by a PR merged in the last {REFS_DAYS} days',
        lambda e: f"#{e['issue']} via #{e['pr']}",
    ),
    "idle_assignees": (
        {"issue", "since"},
        f"Assigned, no activity for {ASSIGNEE_DAYS} days, no open PR",
        lambda e: f"#{e['issue']}, assigned since {e['since']}",
    ),
    "idle_claims": (
        {"issue", "since"},
        f"Claimed in a comment more than {CLAIM_DAYS} days ago, no PR from the claimant",
        lambda e: f"#{e['issue']}, claimed on {e['since']}",
    ),
    "idle_prs": (
        {"issue", "pr", "updated"},
        f"Open PR idle for {PR_DAYS} days, linked to an open issue",
        lambda e: f"#{e['issue']} via #{e['pr']}, last update {e['updated']}",
    ),
    "removed_code": (
        {"issue", "tokens"},
        "Code references no longer on the default branch",
        lambda e: f"#{e['issue']}: " + ", ".join(f"`{t}`" for t in e["tokens"]),
    ),
}

PAGE = "pageInfo { hasNextPage endCursor }"
MERGED_PRS = f"""query($owner: String!, $name: String!, $cursor: String) {{
  repository(owner: $owner, name: $name) {{
    pullRequests(states: MERGED, first: 100, after: $cursor) {{ {PAGE}
      nodes {{ number title body baseRefName mergedAt
        closingIssuesReferences(first: 10) {{ nodes {{ number }} }} }} }} }} }}"""
OPEN_PRS = f"""query($owner: String!, $name: String!, $cursor: String) {{
  repository(owner: $owner, name: $name) {{
    pullRequests(states: OPEN, first: 100, after: $cursor) {{ {PAGE}
      nodes {{ number body updatedAt
        closingIssuesReferences(first: 10) {{ nodes {{ number }} }} }} }} }} }}"""
OPEN_ISSUES = f"""query($owner: String!, $name: String!, $cursor: String) {{
  repository(owner: $owner, name: $name) {{
    issues(states: OPEN, first: 50, after: $cursor) {{ {PAGE}
      nodes {{ number body createdAt updatedAt
        labels(first: 20) {{ nodes {{ name }} }}
        assignees {{ totalCount }}
        comments(last: 20) {{ nodes {{ author {{ login }} authorAssociation createdAt body }} }}
        timelineItems(last: 30, itemTypes: [CROSS_REFERENCED_EVENT, ASSIGNED_EVENT,
                                            REOPENED_EVENT]) {{ nodes {{ __typename
          ... on CrossReferencedEvent {{ createdAt
            source {{ ... on PullRequest {{ author {{ login }} }} }} }}
          ... on AssignedEvent {{ createdAt }}
          ... on ReopenedEvent {{ createdAt }} }} }} }} }} }} }}"""


def graphql_nodes(query, key):
    """Page through one fixed GraphQL connection with the read-only job token."""
    owner, name = os.environ["GITHUB_REPOSITORY"].split("/")
    nodes, cursor = [], None
    while True:
        variables = {"owner": owner, "name": name, "cursor": cursor}
        response = github("/graphql", "POST", {"query": query, "variables": variables})
        if response.get("errors"):
            raise ValueError("GraphQL query failed")
        connection = response["data"]["repository"][key]
        nodes += connection["nodes"]
        if not connection["pageInfo"]["hasNextPage"]:
            return nodes
        cursor = connection["pageInfo"]["endCursor"]


def references(regex, text, unticked_only=False):
    """Return the numbers of this repository's issues referenced in ``text``."""
    repo = os.environ["GITHUB_REPOSITORY"].lower()
    found = set()
    for line in (text or "").splitlines():
        if unticked_only and not UNTICKED.match(line):
            continue
        for m in regex.finditer(line):
            if m.group(1):
                found.add(int(m.group(1)))
            elif m.group(2).lower() == repo:
                found.add(int(m.group(3)))
    return found


def source_index(root):
    """Collect file paths and identifiers of the checked-out package."""
    paths, names = set(), set()
    for file in Path(root).rglob("*.py"):
        paths.add(file.as_posix())
        names.update(re.findall(r"[A-Za-z_]\w*", file.read_text(errors="ignore")))
    return paths, names


def closed_by_merge(merged, default, issues):
    """List open issues that a merged PR links with a closing keyword."""
    found = {}
    for pr in merged:
        merged_at = datetime.fromisoformat(pr["mergedAt"])
        text = f"{pr['title']}\n{pr['body'] or ''}"
        linked = {node["number"] for node in pr["closingIssuesReferences"]["nodes"]}
        for number in sorted(linked | references(KEYWORD_RE, text)):
            issue = issues.get(number)
            if issue is None or number in found or reopened_after(issue, merged_at):
                continue
            if pr["baseRefName"] != default:
                reason = {"reason": "base", "base": pr["baseRefName"]}
            elif number in references(KEYWORD_RE, text, unticked_only=True):
                reason = {"reason": "checklist"}
            else:
                reason = {"reason": "keyword"}
            found[number] = {"issue": number, "pr": pr["number"], **reason}
    return [found[number] for number in sorted(found)]


def reopened_after(issue, when):
    """Check whether the issue was reopened after ``when``, a deliberate reopen."""
    return any(
        event["__typename"] == "ReopenedEvent"
        and datetime.fromisoformat(event["createdAt"]) > when
        for event in issue["timelineItems"]["nodes"]
    )


def referenced(merged, issues, skip, now):
    """List open issues mentioned with "Refs" or similar by a recently merged PR."""
    cutoff = now - timedelta(days=REFS_DAYS)
    pairs = dict.fromkeys(
        (number, pr["number"])
        for pr in merged
        if datetime.fromisoformat(pr["mergedAt"]) > cutoff
        for number in sorted(references(WEAK_RE, pr["body"]))
        if number in issues and number not in skip
    )
    return [{"issue": number, "pr": pr} for number, pr in pairs]


def linked_prs(open_prs, issues):
    """Map open issue numbers to the open PRs that reference them."""
    linked = {}
    for pr in open_prs:
        numbers = {node["number"] for node in pr["closingIssuesReferences"]["nodes"]}
        numbers |= references(KEYWORD_RE, pr["body"]) | references(WEAK_RE, pr["body"])
        for number in numbers & issues.keys():
            linked.setdefault(number, []).append(pr)
    return linked


def idle_assignees(issues, linked, now):
    """List assigned issues with no activity and no open PR."""
    cutoff = now - timedelta(days=ASSIGNEE_DAYS)
    found = []
    for number, issue in issues.items():
        if (
            issue["assignees"]["totalCount"]
            and datetime.fromisoformat(issue["updatedAt"]) < cutoff
            and number not in linked
        ):
            assigned = [
                event["createdAt"][:10]
                for event in issue["timelineItems"]["nodes"]
                if event["__typename"] == "AssignedEvent"
            ]
            since = assigned[-1] if assigned else issue["createdAt"][:10]
            found.append({"issue": number, "since": since})
    return found


def last_claim(issue, now):
    """Return the latest old enough claim comment from a non-maintainer."""
    cutoff = now - timedelta(days=CLAIM_DAYS)
    for comment in reversed(issue["comments"]["nodes"]):
        if (
            comment["author"]
            and comment["authorAssociation"] not in MAINTAINERS
            and datetime.fromisoformat(comment["createdAt"]) < cutoff
            and CLAIM_RE.search(comment["body"])
        ):
            return comment
    return None


def idle_claims(issues, linked, now):
    """List claimed issues with no PR from the claimant and no open PR."""
    found = []
    for number, issue in issues.items():
        claim = last_claim(issue, now)
        if claim is None or number in linked:
            continue
        when = datetime.fromisoformat(claim["createdAt"])
        later_pr = any(
            event["__typename"] == "CrossReferencedEvent"
            and datetime.fromisoformat(event["createdAt"]) > when
            and (event["source"].get("author") or {}).get("login")
            == claim["author"]["login"]
            for event in issue["timelineItems"]["nodes"]
        )
        if not later_pr:
            found.append({"issue": number, "since": claim["createdAt"][:10]})
    return found


def idle_prs(linked, now):
    """List open PRs with no recent update that reference an open issue."""
    cutoff = now - timedelta(days=PR_DAYS)
    return [
        {"issue": number, "pr": pr["number"], "updated": pr["updatedAt"][:10]}
        for number in sorted(linked)
        for pr in linked[number]
        if datetime.fromisoformat(pr["updatedAt"]) < cutoff
    ]


def removed_code(issues, index):
    """List bug reports, or issues citing package paths, whose code is all gone."""
    paths, names = index
    found = []
    for number, issue in issues.items():
        body = issue["body"] or ""
        cited_paths = set(CITED_PATH.findall(body))
        cited_names = {
            name
            for name in (t.split(".")[-1] for t in CITED_NAME.findall(body))
            if "_" in name or re.search(r"[a-z][A-Z]", name)
        }
        is_bug = any(label["name"] == "bug" for label in issue["labels"]["nodes"])
        tokens = sorted(cited_paths) + sorted(cited_names)
        if (
            (is_bug or cited_paths)
            and len(tokens) >= 2
            and not (cited_paths & paths or cited_names & names)
        ):
            found.append({"issue": number, "tokens": tokens[:10]})
    return found


def candidates(merged, default, open_prs, issues, index, now):
    """Build every section from GitHub data and the source index."""
    by_number = {
        issue["number"]: issue for issue in sorted(issues, key=lambda i: i["number"])
    }
    closed = closed_by_merge(merged, default, by_number)
    skip = {entry["issue"] for entry in closed}
    rest = {number: issue for number, issue in by_number.items() if number not in skip}
    linked = linked_prs(open_prs, rest)
    return {
        "closed_by_merge": closed,
        "referenced": referenced(merged, by_number, skip, now),
        "idle_assignees": idle_assignees(rest, linked, now),
        "idle_claims": idle_claims(rest, linked, now),
        "idle_prs": idle_prs(linked, now),
        "removed_code": removed_code(rest, index),
    }


def validate(data):
    """Accept only numbers, dates, branch names and checked code tokens."""
    if type(data) is not dict or set(data) != {"date", "commit", *SECTIONS}:
        raise ValueError("Unexpected result schema")
    if not (_date(data["date"]) and _matches(r"[0-9a-f]{7,40}")(data["commit"])):
        raise ValueError("Invalid run metadata")
    for key, (fields, _, _) in SECTIONS.items():
        entries = data[key]
        if type(entries) is not list or len(entries) > 500:
            raise ValueError("Invalid section")
        for entry in entries:
            if type(entry) is not dict:
                raise ValueError("Invalid entry")
            allowed = fields | ({"base"} if entry.get("reason") == "base" else set())
            if set(entry) != allowed or not all(CHECKS[k](v) for k, v in entry.items()):
                raise ValueError("Invalid entry")
    return data


def render(data):
    """Render the summary issue body from fixed text and validated values."""
    lines = [
        MARKER,
        f"Stale-issue candidates on {data['date']}, default branch at {data['commit'][:8]}. "
        "Candidates only: nothing is closed automatically. Check the evidence before closing.",
    ]
    for key, (_, title, fmt) in SECTIONS.items():
        lines += ["", f"## {title}", ""] + (
            [f"- {fmt(e)}" for e in data[key]] or ["None."]
        )
    return "\n".join(lines) + "\n"


def analyze():
    """Collect stale-issue candidates with read-only queries and no model call."""
    now = datetime.now(UTC)
    default = github("/repos/" + os.environ["GITHUB_REPOSITORY"])["default_branch"]
    data = {
        "date": now.strftime("%Y-%m-%d"),
        "commit": os.environ["GITHUB_SHA"],
        **candidates(
            graphql_nodes(MERGED_PRS, "pullRequests"),
            default,
            graphql_nodes(OPEN_PRS, "pullRequests"),
            graphql_nodes(OPEN_ISSUES, "issues"),
            source_index("pymc_marketing"),
            now,
        ),
    }
    Path("result.json").write_text(json.dumps(validate(data)))


def publish():
    """Create or update this month's bot-authored summary issue."""
    path = Path("result.json")
    if path.stat().st_size > 60000:
        raise ValueError("Oversized result")
    data = validate(json.loads(path.read_text()))
    repo_path = "/repos/" + os.environ["GITHUB_REPOSITORY"]
    title = f"Stale-issue candidates {data['date'][:7]}"
    body = render(data)
    page = 1
    while True:
        items = github(
            f"{repo_path}/issues?state=open&creator=github-actions%5Bbot%5D"
            f"&per_page=100&page={page}"
        )
        for item in items:
            if (
                item["user"]["login"] == "github-actions[bot]"
                and item["title"] == title
                and (item["body"] or "").startswith(MARKER)
            ):
                github(f"{repo_path}/issues/{item['number']}", "PATCH", {"body": body})
                return
        if len(items) < 100:
            break
        page += 1
    github(
        f"{repo_path}/issues",
        "POST",
        {"title": title, "body": body, "labels": ["maintenance"]},
    )


def main():
    """Run the requested workflow phase."""
    phases = {"analyze": analyze, "publish": publish}
    if sys.argv[1:] not in ([name] for name in phases):
        raise ValueError("Unsupported action")
    phases[sys.argv[1]]()


if __name__ == "__main__":
    main()
