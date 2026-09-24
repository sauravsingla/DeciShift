# Concepts

## Decision change

The fundamental object is a record whose final binary decision differs between baseline and candidate.

## Versioned component

A decision pipeline consists of features, model, calibrator, threshold/policy, and deterministic rules. Components should carry stable identity through an explicit version/digest or artifact hash.

## Software-counterfactual attribution

For a changed component set, DeciShift evaluates hybrid pipelines in which components are selected from baseline or candidate. Shapley values allocate the observed software-output difference across those component substitutions. This is not a claim about external causal effects.

## Evidence bundle

A saved run contains record comparisons, attribution, interactions, cohorts, reports and a manifest. The manifest records provenance, run parameters and SHA-256 values for evidence artifacts.

## Decision Contract

A Decision Contract is a user-authored set of deterministic limits for observed behavioral change. DeciShift never invents acceptable limits. Passing a contract means only that those configured checks passed.
