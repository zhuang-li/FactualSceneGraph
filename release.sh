#!/bin/bash

# Release script for FactualSceneGraph
# Usage: ./release.sh [version] [message]

set -e  # Exit on any error

VERSION=${1:-"0.7.0"}
MESSAGE=${2:-"Add DiscoSG-Refiner support"}

echo "🚀 Starting release process for version $VERSION"

# 1. Ensure we're on main branch and up to date
echo "📥 Ensuring we're on main branch and up to date..."
git checkout main
git pull origin main

# 2. Run tests (optional - uncomment if you have tests)
# echo "🧪 Running tests..."
# python -m pytest tests/

# 3. Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# 4. Build the package
echo "📦 Building package..."
python setup.py sdist bdist_wheel

# 5. Check the build
echo "🔍 Checking build..."
python -m twine check dist/*

# 6. Commit changes
echo "💾 Committing changes..."
git add .
git commit -m "Release v$VERSION: $MESSAGE" || echo "No changes to commit"

# 7. Create and push tag
echo "🏷️  Creating and pushing tag..."
git tag -a "v$VERSION" -m "Version $VERSION: $MESSAGE"
git push origin main
git push origin "v$VERSION"

# 8. Upload to PyPI (will prompt for credentials)
echo "📤 Uploading to PyPI..."
echo "⚠️  About to upload to PyPI. Make sure you have your credentials ready!"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m twine upload dist/*
    echo "✅ Successfully uploaded to PyPI!"
else
    echo "❌ Upload cancelled. You can manually upload later with:"
    echo "   python -m twine upload dist/*"
fi

echo "🎉 Release process completed!"
echo "📋 Summary:"
echo "   - Version: $VERSION"
echo "   - Git tag: v$VERSION pushed to origin"
echo "   - PyPI package: uploaded (if confirmed)"
echo ""
echo "🔗 Check your release at:"
echo "   - GitHub: https://github.com/zhuang-li/FACTUAL/releases/tag/v$VERSION"
echo "   - PyPI: https://pypi.org/project/FactualSceneGraph/$VERSION/" 