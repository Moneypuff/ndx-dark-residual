#!/usr/bin/env bash
#
# One-time git history cleanup: purge the large GENERATED artifacts that were committed
# nightly before the site moved to GitHub Pages artifact deployment. These files are all
# rebuilt by CI, so they never need to live in git history.
#
# THIS REWRITES HISTORY. Every commit SHA changes, and pushing requires --force. After you
# push, everyone else must re-clone (or hard-reset), and any open PR/branch must be rebased.
# Run it only AFTER the new Pages workflow is merged and deploying, so no new copies are being
# committed. See the checklist printed at the end.
#
# Usage:
#   scripts/shrink_history.sh            # dry run: clones a mirror, rewrites it, prints sizes
#   scripts/shrink_history.sh --apply    # additionally rewrite THIS working clone in place
#
# Nothing is pushed by this script -- the final `git push --force` is left for you to run
# deliberately.
set -euo pipefail

# Generated paths to remove from ALL history (source files are NOT touched).
PATHS=(
  docs/index.html
  docs/comovement.html
  docs/earnings.html
  earnings_dpi_events.csv
  earnings_dpi_report.html
  earnings_dates_edgar.csv
  earnings_dpi_summary.txt
)

filter_args=()
for p in "${PATHS[@]}"; do filter_args+=(--path "$p"); done

repo_root="$(git rev-parse --show-toplevel)"
origin_url="$(git -C "$repo_root" remote get-url origin)"

echo "==> Dry run: mirroring the repo and rewriting the copy (no changes to your clone)"
work="$(mktemp -d)"
git clone --mirror "$origin_url" "$work/mirror.git" >/dev/null 2>&1 || \
  git clone --mirror "$repo_root/.git" "$work/mirror.git" >/dev/null 2>&1
before=$(du -sh "$work/mirror.git" | cut -f1)
git -C "$work/mirror.git" filter-repo --force "${filter_args[@]}" --invert-paths >/dev/null
git -C "$work/mirror.git" reflog expire --expire=now --all >/dev/null 2>&1 || true
git -C "$work/mirror.git" gc --prune=now --aggressive >/dev/null 2>&1 || true
after=$(du -sh "$work/mirror.git" | cut -f1)
echo "    mirror .git size:  before=$before   after=$after"
rm -rf "$work"

if [[ "${1:-}" == "--apply" ]]; then
  echo "==> Applying to THIS clone in place (history will be rewritten locally, NOT pushed)"
  git -C "$repo_root" filter-repo --force "${filter_args[@]}" --invert-paths
  echo "    Done. filter-repo removed the 'origin' remote as a safety measure."
  cat <<EOF

Next steps (do these deliberately):
  1. Confirm the new Pages workflow is merged to main and has deployed the site.
  2. git -C "$repo_root" remote add origin "$origin_url"
  3. git -C "$repo_root" push --force --all origin
     git -C "$repo_root" push --force --tags origin
  4. Tell any collaborators to re-clone. Rebase or recreate any open PR branches.
EOF
else
  echo "==> Dry run only. Re-run with --apply to rewrite this clone (still no push)."
fi
