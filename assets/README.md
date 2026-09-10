# Assets

This project deliberately does **not** bundle any third-party medical
images or icons, to avoid any copyright/licensing risk (per the project
requirement: *"Use only legally usable/open-license assets"*).

Instead, every visualization in this project (2D disease/symptom charts,
the 2D schematic body-region diagram, 3D PCA/t-SNE feature-space plots,
and the interactive knowledge graph) is **generated programmatically**
with Plotly/NetworkX directly from the dataset - see
`src/visualization/`. This is both license-safe and fully reproducible.

- `images/` - reserved for any future original app branding (e.g. a
  favicon) - currently empty.
- `2d/` - reserved for any future exported static 2D chart images (e.g.
  via `kaleido`) - currently empty; charts are rendered live/interactively
  instead.
- `3d/` - reserved for any future exported static 3D chart images -
  currently empty; charts are rendered live/interactively instead.
