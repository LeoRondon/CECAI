import re
import json

INPUT_FILE = "pdf/cec.txt"
OUTPUT_FILE = "cec_rules.json"

rule_pattern = re.compile(r"^(\d{2}-\d{3}) ([A-Z].*)$")

rules = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

current_rule = None
skip_next_line = False

for i, line in enumerate(lines):
    line = line.rstrip()

    if skip_next_line:
        skip_next_line = False
        continue

    if line.startswith("--`,"):
        continue

    match = rule_pattern.match(line)

    if match:
        if current_rule:
            rules.append(current_rule)

        rule_number = match.group(1)
        title = match.group(2).strip()

        # Handle titles that wrap onto the next line
        if title.endswith("(see") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            title += " " + next_line
            skip_next_line = True

        current_rule = {
            "section": rule_number.split("-")[0],
            "rule": rule_number,
            "title": title,
            "text": []
        }

    elif current_rule:
        current_rule["text"].append(line)

if current_rule:
    rules.append(current_rule)

for rule in rules:
    rule["text"] = "\n".join(rule["text"]).strip()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(rules, f, indent=2, ensure_ascii=False)

print(f"Found {len(rules)} rules")
print(f"Saved to {OUTPUT_FILE}")
