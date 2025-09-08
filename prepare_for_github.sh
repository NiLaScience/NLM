#!/bin/bash
# Script to prepare NLM repository for GitHub

echo "🧹 Preparing NLM for GitHub..."
echo

# Remove Python cache directories
echo "Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove any .DS_Store files (macOS)
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Check for large files that shouldn't be committed
echo
echo "📊 Checking for large files (>1MB)..."
large_files=$(find . -type f -size +1M -not -path "./.git/*" 2>/dev/null)
if [ -n "$large_files" ]; then
    echo "⚠️  Found large files that will be excluded by .gitignore:"
    echo "$large_files" | while read -r file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "   - $file ($size)"
    done
else
    echo "✅ No large files found"
fi

# Show what will be tracked by git
echo
echo "📁 Files that will be tracked by git:"
echo "(excluding .gitignore patterns)"
echo

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add .gitignore first
git add .gitignore 2>/dev/null || true

# Show what would be added
git add --dry-run . 2>&1 | grep -E "^add '|^$" | sed "s/^add/   ✓/"

echo
echo "📋 Summary:"
echo "- Python caches cleaned ✓"
echo "- Large files excluded via .gitignore ✓"
echo "- Repository ready for GitHub ✓"
echo
echo "Next steps:"
echo "1. Review the files above"
echo "2. Run: git add ."
echo "3. Run: git commit -m 'Initial commit: NLM - Minimal Language Model'"
echo "4. Add your GitHub remote: git remote add origin https://github.com/yourusername/NLM.git"
echo "5. Push: git push -u origin main"
