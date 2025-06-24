# Setting Up GitHub Pages for Documentation

## Steps to Enable GitHub Pages

1. **Push the workflow file to GitHub**:
   ```bash
   git add .github/workflows/docs.yml
   git commit -m "Add GitHub Actions workflow for documentation"
   git push origin main
   ```

2. **Enable GitHub Pages in your repository**:
   - Go to your repository on GitHub: https://github.com/KempnerInstitute/alignment
   - Click on "Settings" tab
   - Scroll down to "Pages" in the left sidebar
   - Under "Build and deployment":
     - Source: Select "GitHub Actions"
   - Click "Save"

3. **Trigger the first build**:
   - The workflow will automatically run when you push to main
   - Or manually trigger it:
     - Go to "Actions" tab
     - Select "Deploy Documentation" workflow
     - Click "Run workflow"

4. **Access your documentation**:
   - Once deployed, your documentation will be available at:
   - https://kempnerinstitute.github.io/alignment/

## Documentation URLs in README

The README has been updated with:

1. **Documentation badge**: Points to https://kempnerinstitute.github.io/alignment/
2. **Direct link**: Added under the Documentation section

## Updating Documentation

Whenever you push changes to the `main` branch that affect the documentation:
- The GitHub Actions workflow will automatically rebuild and deploy
- Changes will be live within a few minutes

## Local Documentation Build

To build and preview documentation locally:
```bash
conda activate networkAlignmentAnalysis
python build_docs.py
# Or directly:
cd docs
make html
```

View at: `docs/build/html/index.html` 