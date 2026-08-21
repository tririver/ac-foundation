# ac-document

Host-independent document infrastructure for AC Foundation: immutable source storage,
deterministic parsing, rich-document contracts, cached search, document
structure, and terminology workflows. It contains no paper-provider behavior.

`AcDocumentService` accepts local files and repository artifacts. Academic
identifiers and providers belong to a consumer package, not `ac-document`.

```bash
ac-document --help
ac-document export-rich-document source.md --output-dir publication
python -m pytest packages/ac-document/tests
```
