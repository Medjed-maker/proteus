# Issue tracker: Local Markdown

Issues and PRDs for this repository live as Markdown files in `.scratch/`.

## Conventions

- Use one directory per feature: `.scratch/<feature-slug>/`.
- Store the PRD at `.scratch/<feature-slug>/PRD.md`.
- Store implementation issues at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Record triage state as a `Status:` line near the top of each issue file.
  See `triage-labels.md` for the status strings.
- Append discussion history under a `## Comments` heading at the bottom of
  the file.

## Skill operations

When a skill says to publish to the issue tracker, create the corresponding
file under `.scratch/<feature-slug>/`, creating directories as needed.

When a skill says to fetch a ticket, read the referenced local Markdown file.
The user must provide its path or `<feature-slug>/<NN>` identifier.
