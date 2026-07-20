#!/usr/bin/env bash
#
# Fetch the upstream HiRISE DTM Importer, apply dkogan's patches from
# https://github.com/nasa-jpl/martian/pull/1 (the ones targeting the importer
# itself, not MARTIAN's own code), and zip the result into the location
# MARTIAN expects.
#
# Upstream is pinned to a specific commit for reproducibility. If you bump
# UPSTREAM_COMMIT, re-verify that the patches still apply — the script will
# fail loudly if anything looks off, but silent semantic drift is still
# possible on unrelated code paths.
#
# See:
#   - Upstream:  https://github.com/phaseIV/Blender-Hirise-DTM-Importer
#   - Patches:   https://github.com/nasa-jpl/martian/pull/1
#
set -euo pipefail

REPO_URL="https://github.com/phaseIV/Blender-Hirise-DTM-Importer.git"
UPSTREAM_COMMIT="22f85b60a948983a14271d436dfd670631320bb4"  # phaseIV master as of 2026-11-19
WORKDIR=$(mktemp -d)
DEST_DIR="blender_addons/hirise_dtmimporter"
SRC_DIR="$WORKDIR/hirise_importer/dtmimporter"
REPO_ROOT="$PWD"

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "Cloning upstream HiRISE DTM Importer..."
git clone --quiet "$REPO_URL" "$WORKDIR/hirise_importer"
cd "$WORKDIR/hirise_importer"
git checkout --quiet "$UPSTREAM_COMMIT"
cd "$REPO_ROOT"
echo "  Pinned to commit ${UPSTREAM_COMMIT:0:8}"

echo "Applying dkogan's patches (see nasa-jpl/martian#1)..."
python3 - "$SRC_DIR" <<'PYEOF'
import sys, re, os

src = sys.argv[1]
dtm_path          = os.path.join(src, "mesh", "dtm.py")
importer_path     = os.path.join(src, "ui", "importer.py")
terrainpanel_path = os.path.join(src, "ui", "terrainpanel.py")

def die(msg):
    sys.exit(f"FATAL: {msg}")

# ---- dtm.py: fromthing shim ----
with open(dtm_path) as f:
    content = f.read()

if "fromthing = np.frombuffer" in content:
    print("[dtm.py] fromthing shim already present, skipping.")
else:
    pattern = re.compile(r'^([ \t]*)SPECIAL_VALUES = \{', re.MULTILINE)
    m = pattern.search(content)
    if not m:
        die("[dtm.py] Could not locate 'SPECIAL_VALUES = {' — upstream file structure has changed.")
    indent = m.group(1)
    shim = (
        f"{indent}# To be compatible with both new and old numpy. The new form (frombuffer) is\n"
        f"{indent}# preffered\n"
        f"{indent}try:    fromthing = np.frombuffer\n"
        f"{indent}except: fromthing = np.fromstring\n\n"
    )
    content = pattern.sub(shim + indent + "SPECIAL_VALUES = {", content, count=1)
    content = re.sub(r'np\.(frombuffer|fromstring)\(', 'fromthing(', content)
    print("[dtm.py] fromthing shim applied.")

# ---- dtm.py: nan shim ----
if "nan = np.nan" in content:
    print("[dtm.py] nan shim already present, skipping.")
else:
    nan_pattern = re.compile(
        r'^([ \t]*)(invalid_data_mask = .*\n)([ \t]*)data\[invalid_data_mask\] = np\.(nan|NaN)',
        re.MULTILINE
    )
    m2 = nan_pattern.search(content)
    if not m2:
        die("[dtm.py] Could not locate invalid_data_mask assignment — upstream file structure has changed.")
    indent2 = m2.group(1)
    nan_shim = (
        f"{indent2}# for compatibility with old and new numpy\n"
        f"{indent2}try:    nan = np.nan\n"
        f"{indent2}except: nan = np.NaN\n\n"
    )
    content = nan_pattern.sub(
        nan_shim + m2.group(1) + m2.group(2) + m2.group(3) + "data[invalid_data_mask] = nan",
        content, count=1
    )
    print("[dtm.py] nan shim applied.")

with open(dtm_path, "w") as f:
    f.write(content)

# ---- ui/importer.py and ui/terrainpanel.py: argparse %% fix ----
for name, path in [("importer.py", importer_path), ("terrainpanel.py", terrainpanel_path)]:
    with open(path) as f:
        text = f.read()
    if "100%%" in text:
        print(f"[ui/{name}] argparse %% fix already present, skipping.")
    elif r"100\%" in text:
        text = text.replace(r"100\%", "100%%")
        with open(path, "w") as f:
            f.write(text)
        print(f"[ui/{name}] argparse %% fix applied.")
    else:
        die(f"[ui/{name}] Expected '100\\%' or '100%%' — upstream file structure has changed.")
PYEOF

echo "Verifying patches landed..."
python3 - "$SRC_DIR" <<'PYEOF'
import sys, os
src = sys.argv[1]
checks = [
    (os.path.join(src, "mesh", "dtm.py"),         "fromthing = np.frombuffer"),
    (os.path.join(src, "mesh", "dtm.py"),         "nan = np.nan"),
    (os.path.join(src, "ui", "importer.py"),      "100%%"),
    (os.path.join(src, "ui", "terrainpanel.py"),  "100%%"),
]
failed = []
for path, marker in checks:
    with open(path) as f:
        if marker not in f.read():
            failed.append(f"  MISSING: {marker!r} in {path}")
if failed:
    sys.exit("FATAL: post-patch verification failed:\n" + "\n".join(failed))
print("  All 4 expected markers present.")
PYEOF

echo "Zipping addon..."
cd "$WORKDIR/hirise_importer"
zip -rq dtmimporter.zip dtmimporter/

mkdir -p "$REPO_ROOT/$DEST_DIR"
cp dtmimporter.zip "$REPO_ROOT/$DEST_DIR/"
cd "$REPO_ROOT"

echo "Done. Installed to $DEST_DIR/dtmimporter.zip"
