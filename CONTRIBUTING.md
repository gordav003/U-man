# Contributing and Repository Organization

This document defines a simple workflow to keep `main` stable and to make it
clear from the history why each change was made.

## Branches

`main` is the only permanent branch and must always contain working, reviewed code.
For each completed task, create a short-lived branch from the latest `main`:

- `feature/<short-description>` for new functionality or analysis;
- `fix/<short-description>` for bug fixes;
- `docs/<short-description>` for documentation;
- `refactor/<short-description>` for restructuring without changing behavior;
- `chore/<short-description>` for dependencies and maintenance tasks;
- `agent/<short-description>` for changes prepared by Codex.

Use lowercase letters, English terms, and hyphens, for example
`feature/annual-voltage-duration`. After merging the PR, delete the branch locally and
on GitHub.

## Standard Workflow

1. Update `main` and create a dedicated branch from it.
2. Keep one logical change in a single commit; larger tasks may contain multiple
   separate commits.
3. Run the relevant checks before publishing.
4. Open a pull request against `main` and describe the purpose, impact, and checks
   performed.
5. After review, merge the PR and delete the completed branch.

Do not commit measurement data, credentials, local results, or virtual
environments. These artifacts must remain covered by `.gitignore`.

## Commits and Pull Requests

Keep the title short and use the imperative mood, for example:

- `Add annual voltage duration analysis`
- `Fix missing segment boundaries`
- `Document Parquet smoke test`

A PR should answer four questions: what is changing, why, what is the impact
on the user, and how was the change verified.

## Basic Checks

For Python code changes, at minimum check the syntax:

```powershell
python -m compileall -q .
```

For changes to the Parquet plotter, also run:

```powershell
python parquet_plotter.py --smoke-test
```

For analyses that require measurement files, test them on a limited time range.
In the PR, include the command used and the result, but do not publish confidential data.

## Structure Overview

- `README.md` describes the purpose of the project, installation, and use of the analyses.
- `requirements.txt` is the unified list of Python dependencies.
- `voltage/voltage_data.py` contains shared voltage-data discovery and reading logic.
- `continuous_segments.py` contains shared rules for continuous time segments.
- Analytical scripts are organized into the `correlations`, `measurements`,
  `reactive_power`, and `voltage` packages; move new shared logic into a reusable module.
