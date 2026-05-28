# Two-Axis Channel Information

This project studies a scientific distinction in learned vision networks:
task relevance is not the same as local replaceability.

The reusable implementation lives in the main NodeLens codebase under
`src/nodelens`. The paper-specific scripts, figure generation, manuscript
source, and submission bundles remain in `drafts/alignment_notes`, where the
paper was developed.

## Core Idea

Channel importance mixes two questions:

- Target relevance: what does a channel say about the task?
- Local replaceability: can same-layer peers supply the channel's function if it
  is removed?

The project analyzes these as two axes of channel information. The local axis
uses input capture and peer overlap; the target axis uses task information and
target-excess information. The main finding is that these axes are weakly
aligned after training and have different consequences for removability.

## Code Organization

- Core package: `src/nodelens`
- Generic experiment entry points: `scripts/`
- Paper development area: `drafts/alignment_notes`
- Paper-specific analysis scripts: `drafts/alignment_notes/paper/scripts`
- Paper-specific configs: `drafts/alignment_notes/configs/vision_prune`

This folder is intentionally light. It records the project-level scientific
scope without duplicating paper-generation scripts or manuscript artifacts.
