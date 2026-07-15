import argparse
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile


CLASS_NAME = "SmolVLMVisionEmbeddings"
MODULE_NAME = "transformers.models.smolvlm.modeling_smolvlm"
PATCH_PATH = Path(__file__).with_name("patch.txt")


def load_replacement() -> str:
    patch_text = PATCH_PATH.read_text(encoding="utf-8")
    class_start = patch_text.find(f"class {CLASS_NAME}")
    if class_start == -1:
        raise RuntimeError(f"Could not find {CLASS_NAME} in {PATCH_PATH}")

    replacement = patch_text[class_start:].strip()
    try:
        replacement_tree = ast.parse(replacement)
    except SyntaxError:
        replacement = replacement.removesuffix('"').rstrip()
        replacement_tree = ast.parse(replacement)

    if len(replacement_tree.body) != 1 or not isinstance(replacement_tree.body[0], ast.ClassDef):
        raise RuntimeError(f"Expected one class definition in {PATCH_PATH}")
    if replacement_tree.body[0].name != CLASS_NAME:
        raise RuntimeError(f"Patch defines {replacement_tree.body[0].name}, expected {CLASS_NAME}")
    return replacement + "\n"


def find_target() -> Path:
    spec = importlib.util.find_spec(MODULE_NAME)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Could not locate {MODULE_NAME}; install transformers first")
    return Path(spec.origin)


def find_class(source: str) -> ast.ClassDef:
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {CLASS_NAME} class, found {len(matches)}")
    return matches[0]


def apply_patch(target: Path, check_only: bool) -> bool:
    source = target.read_text(encoding="utf-8")
    replacement = load_replacement()
    class_node = find_class(source)
    source_lines = source.splitlines(keepends=True)
    current_class = "".join(source_lines[class_node.lineno - 1:class_node.end_lineno])

    if current_class.strip() == replacement.strip():
        print(f"Patch already applied: {target}")
        return False
    if check_only:
        print(f"Patch required: {target}")
        return True

    newline = "\r\n" if "\r\n" in source else "\n"
    replacement = replacement.replace("\n", newline)
    updated_source = "".join(
        source_lines[:class_node.lineno - 1]
        + [replacement]
        + source_lines[class_node.end_lineno:]
    )
    ast.parse(updated_source)

    backup = target.with_suffix(target.suffix + ".smolpi.bak")
    if not backup.exists():
        shutil.copy2(target, backup)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(updated_source)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Patched: {target}")
    print(f"Backup: {backup}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the installed Transformers SmolVLM vision embeddings")
    parser.add_argument("--check", action="store_true", help="Check whether the patch is required without changing files")
    parser.add_argument("--target", type=Path, help="Override the modeling_smolvlm.py path")
    args = parser.parse_args()

    target = args.target.resolve() if args.target else find_target()
    changed = apply_patch(target, check_only=args.check)
    if args.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
