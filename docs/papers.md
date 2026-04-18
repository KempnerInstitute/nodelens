# Papers built on this framework

This framework underpins several research projects. Each paper has its own
companion repository with LaTeX source, paper-specific analysis scripts, and
reproducibility instructions. Paper repositories depend on a pinned version
of this framework; the recommended workflow is to clone both, checkout the
framework at the pinned ref, and follow the paper repo's `reproduce.md`.

| Paper | Status | Paper repo | Framework ref | Year |
|-------|--------|------------|---------------|------|
| *Relevance Is Not Replaceability: Orthogonal Axes of Channel Information in Vision Networks* | manuscript companion repo | [KempnerInstitute/alignment_notes](https://github.com/KempnerInstitute/alignment_notes) | `ca438bd1419849775a08d366416486ba2c03ccdc` | 2026 |

## Adding a new paper

When starting a new paper that uses this framework:

1. Create a dedicated git repo for the paper under the Kempner org (e.g.
   `KempnerInstitute/<paper-codename>`).
2. Commit the LaTeX source, figure-generation scripts, paper-specific analysis
   scripts, and a `README.md`, `reproduce.md`, `CITATION.cff`, and
   `pinned_commit.txt` in that repo.
3. The paper repo may live under `drafts/<paper-codename>/` inside this
   framework repo's working tree — `drafts/` is gitignored here, so it won't
   pollute the framework repo. Each paper repo should have its own `.gitignore`.
4. Before a public release, cut a tag in *this* framework repo (e.g.
   `neurips2026-<short-name>-v1`) that matches the framework commit the paper
   used, and record the full SHA in the paper repo's `pinned_commit.txt`.
5. Add a row to the table above.
