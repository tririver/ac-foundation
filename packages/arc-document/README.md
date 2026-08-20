# arc-document

Host-independent document infrastructure for ARC: immutable source storage,
deterministic parsing, rich-document contracts, cached search, document
structure, and terminology workflows. It contains no paper-provider behavior.

`ArcDocumentService` accepts local files and repository artifacts. Academic
identifiers and providers belong to `arc-paper`.

```bash
arc-document --help
python -m pytest packages/arc-document/tests
```
