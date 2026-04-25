# Project Workflows

This directory contains applied workflows built on top of the reusable
`nodelens` package. Each project folder should explain what analysis it runs,
which configs are relevant, what outputs it produces, and how to inspect or
regenerate those outputs.

Reusable code belongs in `src/nodelens/`. Project folders should stay focused
on reproducible usage: configs, small helper scripts, artifact descriptions,
and project-specific notes that help readers connect the public results to the
shared library.

## Projects

- `supernodes_scar/`: workflow for the Supernodes and SCAR analysis of
  loss-sensitive FFN channels in LLMs.
