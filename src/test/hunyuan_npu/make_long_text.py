"""Flatten the LoCoMo conversations into one plain-text file for test_long.

Usage: python3 make_long_text.py <locomo10.json> <out.txt>
"""
import json
import sys


def session_numbers(conversation):
    numbers = []
    for key in conversation:
        parts = key.split("_")
        if len(parts) == 2 and parts[0] == "session":
            numbers.append(int(parts[1]))
    return sorted(numbers)


def conversation_lines(conversation):
    lines = []
    for n in session_numbers(conversation):
        date = conversation.get(f"session_{n}_date_time", "")
        lines.append(f"[Session {n}, {date}]")
        for turn in conversation[f"session_{n}"]:
            lines.append(f"{turn['speaker']}: {turn['text']}")
        lines.append("")
    return lines


def main():
    source, target = sys.argv[1], sys.argv[2]
    with open(source, encoding="utf-8") as f:
        samples = json.load(f)
    lines = []
    for sample in samples:
        lines.extend(conversation_lines(sample["conversation"]))
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{len(samples)} conversations, {len(lines)} lines -> {target}")


if __name__ == "__main__":
    main()
