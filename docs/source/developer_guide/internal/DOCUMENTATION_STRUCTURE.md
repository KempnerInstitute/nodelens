# Documentation Structure

This document describes the organization of documentation in the alignment framework.

## Documentation Locations

### Main Documentation (`docs/`)
Central documentation hub containing:
- User guides
- Developer guides  
- API references
- Examples and tutorials

Structure:
```
docs/
├── source/
│   ├── index.md                    # Main documentation index
│   ├── ALIGNMENT_MODULE_GUIDE.md   # Complete user guide
│   ├── METRICS_REFERENCE.md        # Mathematical metric descriptions
│   ├── ALL_METRICS_LIST.md         # Quick metric reference
│   ├── user_guide/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── pruning_strategies.md   # Comprehensive pruning guide
│   ├── developer_guide/
│   │   └── architecture.md         # Framework architecture
│   ├── api/                        # API documentation
│   └── examples/                   # Example notebooks
└── CODEBASE_ORGANIZATION.md        # Codebase structure guide
```

### Module Documentation (`src/alignment/*/README.md`)
Each module contains its own README with:
- Module purpose and overview
- Available functionality
- Usage examples
- Best practices
- Integration with other modules

Modules with documentation:
- `analysis/` - Result aggregation and reporting
- `configs/` - Configuration management
- `data/` - Dataset handling
- `experiments/` - Experiment framework
- `external/` - External dependencies
- `infrastructure/` - System utilities
- `pruning/` - Pruning strategies
- `training/` - Training utilities

### Project Root Documentation
- `README.md` - Main project overview and quick start
- `CHANGELOG.md` - Version history
- `LICENSE` - License information

## Documentation Guidelines

### Module READMEs
Each module README should include:
1. **Overview** - What the module does
2. **Structure** - File organization
3. **Key Components** - Main classes/functions
4. **Usage Examples** - How to use the module
5. **Integration** - How it works with other modules
6. **Best Practices** - Recommendations

### User Guides
User-facing documentation should:
- Use clear, simple language
- Include working code examples
- Explain concepts before code
- Provide troubleshooting tips

### Developer Guides
Developer documentation should:
- Explain design decisions
- Document architecture patterns
- Include contribution guidelines
- Describe testing approaches

## Recent Changes

### Documentation Consolidation (Latest)
1. Moved `ARCHITECTURE_GUIDE.md` → `docs/source/developer_guide/architecture.md`
2. Moved `PRUNING_STRATEGIES_GUIDE.md` → `docs/source/user_guide/pruning_strategies.md`
3. Removed duplicate `src/alignment/README.md`
4. Created comprehensive `docs/source/index.md`
5. Updated main README with proper links

### Benefits
- Centralized documentation discovery
- Clear separation of user vs developer docs
- Module-specific docs stay with code
- Reduced duplication
- Better organization

## Accessing Documentation

### For Users
Start with:
1. Main README for overview
2. `docs/source/index.md` for comprehensive docs
3. Module READMEs for specific functionality

### For Developers
Key resources:
1. `docs/source/developer_guide/architecture.md` - Design principles
2. `docs/CODEBASE_ORGANIZATION.md` - Code structure
3. Module READMEs - Implementation details

### Online Documentation
Full documentation is available at: https://kempnerinstitute.github.io/alignment/ 