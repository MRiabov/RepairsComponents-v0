#!/usr/bin/env bash
# Run Ruff on staged files only
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$')
if [ -z "$staged_files" ]; then
    exit 0
fi

# Run Ruff format
ruff format $staged_files

# Stage the fixed files
git add $staged_files
