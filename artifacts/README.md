# Complete experiment artifacts

The scientific snapshot contains **3,896 original server files** (26,367,726,620 bytes).
Of these, 2,873 source, configuration, log and result files are tracked directly in Git.
The remaining 1,023 files (26,350,356,941 bytes before archive compression) are catalogued
as 27 independent archives, separate from the source checkout.
They include all server checkpoints, prediction arrays, prepared arrays, historical
experiment bundles and compatibility wheels. Search, diagnostic and smoke runs are
retained alongside selected runs; the experiment index identifies their roles.

`server_manifest.json` records every original file, its original relative path, byte size,
SHA-256 digest and storage location. Its `omissions` list covers only regenerable Python
bytecode, package metadata, runtime caches and lock files. Additional source and analysis
materials have separate provenance in `docs/BOOTSTRAP_PROVENANCE.json`.

When archive parts have been provided, each can be independently extracted into the
repository root. For archive and per-file verification, use Python 3.12+:

```bash
python scripts/restore_artifacts.py --archive-dir PATH_TO_ARCHIVE_PARTS
python scripts/restore_artifacts.py --verify-only
```

To restore a subset, use e.g. `--part trefin-artifacts-01.tar.gz`. Consult the manifest
to find the part containing a particular checkpoint or prediction file. The restore
script verifies each archive and extracts it safely. Archives in `--archive-dir` are
retained. An explicitly supplied download location may instead be passed with
`--base-url URL`; in that mode, the script downloads one archive at a time and removes
only its own temporary download after extraction. Allow at least 28 GB of additional
free space for a full restore.

`release_assets.json` records archive filenames, sizes, member counts and SHA-256 values.
It intentionally contains no link to an author account or its hosting pages. The compact
analysis files and results in the source checkout do not require these large archives.
File hashes are preserved without newline conversion through the root `.gitattributes`.
Scientific values and original server records are preserved.

## Historical transport bundle

The original `server_bundle.tar.gz` is an incomplete historical transfer archive
(an existing gzip/tar truncation). It is retained byte-for-byte for provenance, but
is not a restoration source. Use the 27 verified release archives and the current
working-tree source instead. The other original historical bundles retain their
original contents and names.
