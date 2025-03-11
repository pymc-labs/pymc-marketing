"""This script parses the github action log for test warnings."""

import re
import sys
from pathlib import Path

start_pattern = re.compile(r"==== warnings summary")
end_pattern = re.compile(r"capture-warnings.html")

# The leading marker of the log times which are to be trimmed off
# Example datetime 2025-03-10T21:38:45.5291113Z
datetime_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d*Z ")


def extract_lines(lines: list[str]) -> list[str]:
    warnings = []

    in_section = False
    for line in lines:
        detect_start = start_pattern.search(line)
        detect_end = end_pattern.search(line)

        if detect_start:
            in_section = True

        if in_section:
            warnings.append(line)

        if not detect_start and in_section and detect_end:
            break

    return warnings[:-1]


def trim_up_to_match(pattern, string: str) -> str:
    match = pattern.search(string)
    if not match:
        return ""

    return string[match.start() :]


def trim_through_match(pattern, string: str) -> str:
    match = pattern.search(string)
    if not match:
        return ""

    return string[match.end() :]


def trim(pattern, lines: list[str], including: bool = False) -> list[str]:
    _trim = trim_through_match if including else trim_up_to_match
    return [_trim(pattern, line) for line in lines]


def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def format_warnings(warnings: list[str]) -> list[str]:
    return trim(end_pattern, warnings[:1]) + trim(
        datetime_pattern, warnings[1:-1], including=True
    )


def read_lines_from_stdin():
    return sys.stdin.read().splitlines()


def read_from_file(file: Path):
    return file.read_text().splitlines()


def main(read_lines):
    lines = read_lines()
    times = extract_lines(lines)
    parsed_times = format_warnings(times)
    print("\n".join(parsed_times))


if __name__ == "__main__":
    read_lines = read_lines_from_stdin
    main(read_lines)
