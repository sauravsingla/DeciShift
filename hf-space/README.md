---
title: DeciShift — Decision Change Explorer
emoji: 🔀
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Interactive ML decision-change regression explorer
---

# DeciShift — Decision Change Explorer

Interactive browser-only explorer for DeciShift's public Digits/XGBoost decision-change case study.

It visualizes how a candidate ML decision system can improve model-level metrics while individual operational actions still move, how those transitions are attributed across versioned software components, and how a Decision Contract can block a release when a governed cohort exceeds its declared limit.

The displayed evidence is generated from the public DeciShift repository before publishing the Space. The Space itself does **not** execute an ML model, external tool, or hosted API.

- Source: https://github.com/sauravsingla/DeciShift
- Dataset: https://huggingface.co/datasets/sauravsingla08/DeciShift-Decision-Change-Benchmark
- PyPI: https://pypi.org/project/decishift/

Apache-2.0. Upstream source-data terms remain applicable to the original scikit-learn Digits dataset.
