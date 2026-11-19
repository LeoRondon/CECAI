"""Parse the CEC plain-text export (pdf/cec.txt) into cec_rules.json."""

import json
import re

from cec.config import settings

INPUT_FILE = settings.cec_text_file
OUTPUT_FILE = settings.rules_file

# Rule numbers have a 1- or 2-digit section (0, 2, 4, 6, 8, 10, 12 ... 86)
rule_pattern = re.compile(r"^(\d{1,2}-\d{3}) ([A-Z].*)$")

# Section 0 has no numbered rules: it is "Object", "Scope" and a list of
# "Term — meaning" definitions. These patterns let us index it too.
SECTION0_START = "Object (see Appendix B)"
section0_heading_pattern = re.compile(r"^(Object|Scope|Definitions)( \(see Appendix B\))?$")
definition_pattern = re.compile(r"^([A-Z][^—]{0,100}?) —( .*)?$")

rules = []

with open(INPUT_FILE, encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

current_rule = None
skip_next_line = False
in_section0 = False
in_definitions = False


def new_entry(section, rule, title):
    return {"section": section, "rule": rule, "title": title, "text": []}


for i, line in enumerate(lines):
    line = line.rstrip()

    if skip_next_line:
        skip_next_line = False
        continue

    if line.startswith("--`,"):
        continue

    match = rule_pattern.match(line)

    if match:
        in_section0 = False
        if current_rule:
            rules.append(current_rule)

        rule_number = match.group(1)
        title = match.group(2).strip()

        # Handle titles that wrap onto the next line
        if title.endswith("(see") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            title += " " + next_line
            skip_next_line = True

        current_rule = new_entry(rule_number.split("-")[0], rule_number, title)
        continue

    # Section 0 begins at the body's "Object" heading and ends at the first
    # numbered rule (2-000).
    if not rules and current_rule is None and line == SECTION0_START:
        in_section0 = True

    if in_section0:
        heading = section0_heading_pattern.match(line)
        definition = definition_pattern.match(line) if in_definitions else None

        if heading:
            if current_rule:
                rules.append(current_rule)
            name = heading.group(1)
            in_definitions = name == "Definitions"
            current_rule = new_entry("0", "Section 0", name)
            continue

        if definition and not line.startswith("Section "):
            if current_rule:
                rules.append(current_rule)
            term = definition.group(1).strip()
            current_rule = new_entry("0", "Section 0", f"Definition: {term}")
            rest = (definition.group(2) or "").strip()
            current_rule["text"].append(f"{term} — {rest}".rstrip(" —"))
            continue

    if current_rule:
        current_rule["text"].append(line)

if current_rule:
    rules.append(current_rule)

for rule in rules:
    rule["text"] = "\n".join(rule["text"]).strip()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(rules, f, indent=2, ensure_ascii=False)

print(f"Found {len(rules)} rules")
print(f"Saved to {OUTPUT_FILE}")
