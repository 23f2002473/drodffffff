#!/bin/bash

echo "=== Starting Cleanup ==="

# 1. Remove old snap revisions
echo ">>> Cleaning old Snap revisions..."
snap list --all | awk '/disabled/{print $1, $3}' | \
while read snapname revision; do
    echo "Removing $snapname revision $revision..."
    sudo snap remove "$snapname" --revision="$revision"
done

# 2. Reduce snap retained versions (keep only 2 going forward)
echo ">>> Setting snap retain to 2 revisions..."
sudo snap set system refresh.retain=2

# 3. Clean conda cache
if command -v conda &> /dev/null; then
    echo ">>> Cleaning conda cache..."
    conda clean --all --yes
else
    echo ">>> Conda not found, skipping..."
fi

# 4. Clean pip cache
if command -v pip &> /dev/null; then
    echo ">>> Cleaning pip cache..."
    pip cache purge
else
    echo ">>> Pip not found, skipping..."
fi

# 5. Clear user cache (~/.cache)
echo ">>> Clearing user cache..."
rm -rf ~/.cache/*

# 6. Clean old system logs
echo ">>> Cleaning old system logs (keeping last 7 days)..."
sudo journalctl --vacuum-time=7d

echo "=== Cleanup Complete ==="
echo "Disk usage now:"
df -h

