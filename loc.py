import os

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


def is_valid_py_file(path: str) -> bool:
    return path.endswith(".py") and os.path.isfile(path)


def count_loc_in_file(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and not line.strip().startswith("#"))


def count_loc_in_project(root_dir: str = ".") -> int:
    total_loc = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if is_valid_py_file(file_path):
                total_loc += count_loc_in_file(file_path)
    return total_loc


if __name__ == "__main__":
    loc = count_loc_in_project()
    print(f"Total lines of code: {loc}")
# 7.16 10.5k.
