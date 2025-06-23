# Building HTML Documentation

This guide explains how to build HTML documentation from the markdown and RST files in this project.

## Prerequisites

Install the required documentation packages:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-copybutton sphinx-autobuild
```

Or using the requirements file:

```bash
pip install -r docs/requirements-docs.txt
```

## Documentation Structure

The documentation uses Sphinx with the following structure:

```
docs/
├── source/              # Sphinx source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main documentation index
│   ├── api/            # API documentation
│   ├── user_guide/     # User guides
│   └── ...
├── build/              # Built HTML output (generated)
├── Makefile           # Build commands
└── *.md               # Markdown documentation files
```

## Building HTML Documentation

### Method 1: Using Make (Linux/Mac)

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# Clean previous builds
make clean

# Build with auto-reload for development
make livehtml
```

### Method 2: Direct Sphinx Commands

From the `docs/` directory:

```bash
# Build HTML
sphinx-build -b html source build/html

# Build with specific options
sphinx-build -b html -W source build/html  # Treat warnings as errors

# Auto-build with live reload
sphinx-autobuild source build/html
```

### Method 3: From Project Root

```bash
# Build documentation from project root
cd docs && make html

# Or using Python
python -m sphinx -b html docs/source docs/build/html
```

## Including Markdown Files

To include the new markdown documentation files in the HTML build:

1. **Add to index.rst**:
   ```rst
   .. toctree::
      :maxdepth: 2
      :caption: Metrics Documentation
      
      ../ALIGNMENT_MODULE_GUIDE
      ../METRICS_REFERENCE
      ../METRICS_IMPLEMENTATION_DETAILS
      ../ALL_METRICS_LIST
   ```

2. **Or create a dedicated metrics section** in `docs/source/metrics.rst`:
   ```rst
   Metrics Documentation
   ====================
   
   .. include:: ../../METRICS_REFERENCE.md
      :parser: myst_parser.sphinx_
   ```

## Viewing the Documentation

After building:

1. **Local viewing**:
   ```bash
   # Open in default browser
   open docs/build/html/index.html  # Mac
   xdg-open docs/build/html/index.html  # Linux
   start docs/build/html/index.html  # Windows
   ```

2. **Python HTTP server**:
   ```bash
   cd docs/build/html
   python -m http.server 8000
   # Then visit http://localhost:8000
   ```

3. **Live development server**:
   ```bash
   cd docs
   make livehtml
   # Auto-opens at http://127.0.0.1:8000
   ```

## Customizing the Build

### Theme Options

Edit `docs/source/conf.py`:

```python
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}
```

### Adding Custom CSS

1. Create `docs/source/_static/custom.css`
2. Add to `conf.py`:
   ```python
   html_static_path = ['_static']
   html_css_files = ['custom.css']
   ```

### Math Rendering

The configuration already includes MathJax for rendering mathematical expressions:
- Inline math: `$...$` or `\(...\)`
- Display math: `$$...$$` or `\[...\]`

## Deploying Documentation

### GitHub Pages

1. Build documentation:
   ```bash
   cd docs
   make html
   ```

2. Copy to gh-pages branch:
   ```bash
   git checkout gh-pages
   cp -r docs/build/html/* .
   git add -A
   git commit -m "Update documentation"
   git push origin gh-pages
   ```

### ReadTheDocs

1. Connect repository to ReadTheDocs
2. Configuration is already set up in `docs/source/conf.py`
3. Documentation builds automatically on push

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure project is in Python path
   ```python
   # In conf.py
   sys.path.insert(0, os.path.abspath('../..'))
   ```

2. **Markdown not rendering**: Install myst-parser
   ```bash
   pip install myst-parser
   ```

3. **Missing dependencies**: Install all requirements
   ```bash
   pip install -r docs/requirements-docs.txt
   ```

4. **Build warnings**: Use strict mode to debug
   ```bash
   make html SPHINXOPTS="-W"
   ```

## Development Workflow

1. **Edit markdown files** in `docs/`
2. **Run live server**: `make livehtml`
3. **View changes** in browser (auto-reloads)
4. **Commit changes** when satisfied

## Adding New Documentation

1. **Create markdown file**: `docs/NEW_TOPIC.md`
2. **Add to index**: Edit `docs/source/index.rst`
3. **Build and verify**: `make html`
4. **Check links**: `make linkcheck` 