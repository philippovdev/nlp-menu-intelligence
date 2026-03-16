# Report Checklist

## Abstract

Must include:

- problem summary
- dataset summary
- model or pipeline summary
- main quantitative result
- repository link

## Introduction

Must include:

- why menu structuring is useful
- why this is an NLP problem
- what is unique about the project
- clear list of contributions

## Team

If the project is solo, state that all work was done by one person and list the
main responsibilities.

## Related Work

Must include:

- direct or near-direct comparators
- at least one comparison table
- references for every claimed external result

Evidence to prepare:

- filled matrix from `docs/course/related-work.md`
- bibliography entries in `report/lit.bib`

## Model Description

Must include:

- full pipeline diagram
- line normalization and parsing stages
- classification stage
- extraction stage
- aggregation into JSON

Figures to prepare:

- pipeline figure
- optional contract or schema figure

## Dataset

Must include:

- data source description
- collection policy
- annotation format
- split policy
- statistics table

Figures and tables to prepare:

- dataset statistics table
- class distribution chart
- optional source type chart

## Experiments

Must include:

- metrics
- experiment setup
- baselines
- hyperparameters or major configuration choices

Tables to prepare:

- experiment setup summary
- baseline configuration summary

## Results

Must include:

- final comparison table
- interpretation of results
- error analysis
- sample outputs

Figures and tables to prepare:

- main results table
- confusion matrix
- output samples table

## Conclusion

Must include:

- what was built
- what was measured
- what worked best
- what remains future work

## Submission Checklist

- repository is public and readable
- report PDF exists
- repository link is placed in the report
- reproducibility instructions are present in the repo
- final results table is filled
- dataset description is complete
- live smoke is completed using `docs/course/live-smoke-checklist.md`
