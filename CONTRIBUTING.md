# Contributing to Mouse Vision

Thank you for your interest in contributing! This document outlines the workflow and expectations for code contributions.

## Getting Started

1. **Fork and clone** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature-name`
3. **Make changes** following the guidelines below
4. **Test thoroughly** before submitting
5. **Open a Pull Request** with a clear description

---

## Code Guidelines

### Style & formatting

- **Python version:** 3.10+
- **Indentation:** 4 spaces (PEP 8)
- **Line length:** ≤100 characters preferred, ≤120 maximum
- **Type hints:** Required for function arguments and returns (Python 3.10+)

Example:
```python
def compute_interaction_feature(
    pose_records: list[dict],
    window_size: int = 5,
) -> list[float]:
    """Compute temporal interaction features from pose sequence.
    
    Args:
        pose_records: List of canonical pose records
        window_size: Frames to combine in rolling window
        
    Returns:
        Interaction feature values (one per input record)
    """
    features = []
    for i, record in enumerate(pose_records):
        # Implementation...
        features.append(value)
    return features
```

### Documentation

- Add docstrings to all public functions
- Document complex logic with inline comments
- Update `DEVELOPER.md` if adding new scripts or workflows
- Reference `data_contract.json` when working with data formats

### File organization

- **Pipeline scripts:** Scripts that modify data go in `scripts/` with clear naming
  - `scripts/my_stage_name.py` for a new pipeline stage
  - `scripts/my_utility.py` for helper functions
- **Dashboard code:** Interactive UI code stays in `app/`
- **Configuration:** All tunable parameters in `configs/*.json`, not hardcoded

---

## Making Changes

### Adding a new feature

1. **Edit `scripts/build_features.py`**
   ```python
   def add_my_feature(df: pd.DataFrame) -> pd.DataFrame:
       """Compute my_feature from pose data."""
       df["my_feature"] = compute(df[["b_nose_x", "w_nose_x"]])
       return df
   ```

2. **Register in `ALL_FEATURE_COLS`** if the model should use it

3. **Test on sample data**
   ```bash
   python scripts/build_features.py --config configs/mvp_config.json
   ```

4. **Check output:** `data/eda_outputs/features_top_view.csv`
   - No NaN explosion
   - Values are in reasonable ranges
   - Data types are consistent

### Modifying the model

1. **Keep training/test split logic identical** to avoid data leakage
2. **Document new hyperparameters** in the function docstring
3. **Log model performance metrics** in `train_eval.py`
4. **Update** `baseline_model_summary.json` schema if adding new metrics

### Updating the dashboard

1. **Test locally** before committing
   ```bash
   streamlit run app/scientist_dashboard.py
   ```

2. **Ensure responsive design** — test on mobile widths (320px, 768px, 1024px)
3. **Cache expensive operations** with `@st.cache_data`
4. **Add user feedback** for long operations (e.g., progress bars)

---

## Testing

### Minimal test before committing

```bash
# Syntax check
python -m py_compile scripts/my_script.py app/scientist_dashboard.py

# Config validation
python -m json.tool configs/mvp_config.json > /dev/null

# Quick pipeline run on small dataset
python scripts/run_pipeline.py --config configs/mvp_config.json

# Verify outputs exist and are valid
ls -la data/eda_outputs/batch_predictions.csv
python -c "import pandas as pd; df = pd.read_csv('data/eda_outputs/batch_predictions.csv'); assert len(df) > 0, 'No predictions'"
```

### After changing features or model

1. **Re-train:**
   ```bash
   python scripts/train_eval.py --config configs/mvp_config.json
   ```

2. **Check metrics:**
   ```bash
   python -c "import json; m = json.load(open('data/eda_outputs/baseline_model_summary.json')); print(f\"AUC: {m['auc']:.3f}\")"
   ```

3. **Validate predictions:**
   ```bash
   python scripts/predict_batch.py --config configs/mvp_config.json
   python -c "import pandas as pd; df = pd.read_csv('data/eda_outputs/batch_predictions.csv'); assert (df['y_proba_close'] >= 0).all() and (df['y_proba_close'] <= 1).all()"
   ```

---

## Commit Messages

Use clear, descriptive commit messages:

```
Good:
  feat: Add nose_acceleration feature to interaction modeling
  fix: Exclude frames with corrupted pose data from segments
  docs: Update DEVELOPER.md with feature engineering guide
  refactor: Consolidate pose validation logic in validators.py
  perf: Cache pose index loading in dashboard

Avoid:
  Updated stuff
  Fixed bug
  WIP
```

Format:
```
<type>: <subject>

<optional body explaining why change was needed>
```

Types:
- `feat:` New feature or capability
- `fix:` Bug fix or regression
- `docs:` Documentation updates
- `refactor:` Code reorganization (no behavior change)
- `perf:` Performance improvement
- `test:` Test coverage additions
- `chore:` Dependency or tooling updates

---

## Pull Request Process

### Before opening a PR

1. **Update your branch** with latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Test one more time:**
   ```bash
   python scripts/run_pipeline.py --config configs/mvp_config.json
   streamlit run app/scientist_dashboard.py
   ```

3. **Check what changed:**
   ```bash
   git diff origin/main
   ```

### PR description template

```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed?

## Changes
- [ ] Feature X added to `scripts/build_features.py`
- [ ] Model retraining: new AUC is X.XXX
- [ ] Updated `DEVELOPER.md` section on feature engineering
- [ ] Added validation for [data constraint]

## Testing
- Ran full pipeline: ✓
- Dashboard tested: ✓
- Edge cases checked: [describe]

## Related issues
Closes #123
```

### Code review expectations

- Reviewers may ask clarifying questions about **why** changes were made
- Be prepared to iterate — multiple rounds are normal
- Small, focused PRs are reviewed faster than large ones
- Automated checks (syntax, JSON validity) must pass

---

## Common Mistakes to Avoid

❌ **Don't:**
- Hardcode file paths (`/home/user/data` instead of `data/`)
- Commit large artifacts (`*.mp4`, `*.joblib`, `*.h5`)
- Ignore `.gitignore` — respect checked-in exclusions
- Change config syntax — maintain backward compatibility
- Merge without testing

✅ **Do:**
- Use `Path` for cross-platform compatibility
- Reference config paths dynamically
- Test on multiple datasets
- Document breaking changes prominently
- Update run tests after changes

---

## Troubleshooting Reviews

**"Tests are failing"**
- Run `python scripts/run_pipeline.py` locally to reproduce
- Check console output for specific error messages
- Share error output in PR comment

**"Merge conflict"**
- Run `git fetch origin && git rebase origin/main`
- Manually resolve conflicts in your editor
- Test again after resolving

**"Review taking a long time"**
- Comment with an update or rebase if there are new commits
- Break into smaller PRs if scope is large
- Ask for guidance if feedback is unclear

---

## Questions?

- **System design questions:** Read [docs/system-architecture.md](docs/system-architecture.md)
- **Development setup:** See [DEVELOPER.md](DEVELOPER.md)
- **Project goals:** Check [README.md](README.md)

Thank you for contributing! 🎉
