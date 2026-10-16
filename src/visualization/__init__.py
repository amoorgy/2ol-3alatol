"""Figures for the d5 Cohere track.

Layering rule, enforced by a test: `theme`, `loaders` and `series` are stdlib
only -- they turn results JSON into figure-ready values and are testable
without the venv. The matplotlib dependency is confined to `figures_*` and
`render_all`.
"""
