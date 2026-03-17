#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <new-version>"
  echo "Example: $0 0.1.0-alpha.1"
  exit 1
fi

NEW_VERSION="$1"

# Extract current version from workspace root
CURRENT_VERSION=$(grep '^version' "$WORKSPACE_ROOT/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')

if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
  echo "Already at version $NEW_VERSION"
  exit 0
fi

echo "Bumping $CURRENT_VERSION -> $NEW_VERSION"

# 1. Update workspace version
sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$WORKSPACE_ROOT/Cargo.toml"

# 2. Update all path dependency version pins
find "$WORKSPACE_ROOT" -name Cargo.toml -not -path '*/target/*' -not -path '*/submodules/*' \
  -exec sed -i "s/version = \"=$CURRENT_VERSION\"/version = \"=$NEW_VERSION\"/g" {} +

echo "Updated $(grep -rl "=$NEW_VERSION" "$WORKSPACE_ROOT"/*/Cargo.toml | wc -l) crate Cargo.toml files"
echo "Done. Run 'cargo check' to verify."
