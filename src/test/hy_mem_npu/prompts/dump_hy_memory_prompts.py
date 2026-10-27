"""Copy the hy_memory engine's prompt templates into hy_memory_prompts.json.

The chain test (test_hy_mem --chain) runs the engine's real EXTRACT and RECONCILE
prompts, which live as string constants in the Hy-Memory sources. This writes them
to one JSON file so the C++ test needs no Python at run time.

Trick: ast.parse() only turns the file text into a syntax tree -- it does not
execute anything -- so we get the exact strings without importing hy_memory or its
dependencies (the same way models_simple/verify_hy_mem_30A3B.py reads them).

Usage: python3 dump_hy_memory_prompts.py [path to the Hy-Memory clone]
"""
import ast
import json
import os
import sys

HY_MEMORY_SRC = sys.argv[1] if len(sys.argv) > 1 else "/scratch/michyu/Projects/hy_memory_src/Hy-Memory"

# key in the json -> (file in the Hy-Memory clone, constant name)
PROMPTS = {
    "extract_system": ("hy_memory/agent/extractor.py", "EXTRACT_SYSTEM_PROMPT"),
    "extract": ("hy_memory/agent/extractor.py", "EXTRACT_PROMPT"),
    "reconcile": ("hy_memory/agent/reconciler.py", "RECONCILE_PROMPT"),
}


def load_prompt(rel_path, name):
    """Return the string constant assigned to `name` at the top level of a file."""
    with open(os.path.join(HY_MEMORY_SRC, rel_path)) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return node.value.value
    raise KeyError(f"{name} not found in {rel_path}")


out = {key: load_prompt(path, name) for key, (path, name) in PROMPTS.items()}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy_memory_prompts.json")
with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {out_path}: " + ", ".join(f"{k} {len(v)} chars" for k, v in out.items()))
