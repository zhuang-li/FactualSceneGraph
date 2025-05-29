# 📦 Release Guide for FactualSceneGraph

## 🚀 Quick Release (Automated)

Use the provided release script:

```bash
# Default release
./release.sh

# Custom version/message
./release.sh "0.6.1" "Add multi-sentence scene graph parsing"
```

## 📝 Manual Release Process

### Prerequisites
```bash
pip install twine build wheel
```

### Step-by-Step Commands

```bash
# 1. Ensure you're on main branch and up to date
git checkout main
git pull origin main

# 2. Clean previous builds
rm -rf build/ dist/ *.egg-info/

# 3. Build the package
python setup.py sdist bdist_wheel

# 4. Check the build quality
python -m twine check dist/*

# 5. Commit and push changes
git add .
git commit -m "Release v0.6.1: Add multi-sentence scene graph parsing"
git push origin main

# 6. Create and push git tag
git tag -a "v0.6.1" -m "Version 0.6.1: Add multi-sentence scene graph parsing"
git push origin v0.6.1

# 7. Upload to PyPI
python -m twine upload dist/*

# 8. Verify the release
pip install --upgrade FactualSceneGraph
```

## 🔧 PyPI Configuration

Create `~/.pypirc` for authentication:

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-your-api-token-here
```

## ✅ Post-Release Checklist

- [ ] Check [PyPI package page](https://pypi.org/project/FactualSceneGraph/)
- [ ] Verify installation: `pip install --upgrade FactualSceneGraph`
- [ ] Test the new features work as expected
- [ ] Create GitHub Release with release notes
- [ ] Update documentation if needed
- [ ] Announce the release (if applicable)

## 🐛 Troubleshooting

### Common Issues:

1. **Authentication errors**: Check your PyPI token in `~/.pypirc`
2. **Version conflicts**: Ensure version number is incremented
3. **Build errors**: Check for syntax errors in setup.py
4. **Git push errors**: Ensure you have push permissions to the repository

### Testing Before Release:

```bash
# Test local installation
pip install -e .

# Test import
python -c "from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser; print('Import successful')"

# Test new feature
python -c "
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', parser_type='sentence_merge')
print('Sentence merge parser created successfully')
"
``` 