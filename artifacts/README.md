# Complete experiment artifacts

The scientific snapshot contains **3,896 original server files** (26,367,726,620 bytes).
Of these, 2,873 source, configuration, log and result files are tracked directly in Git.
The remaining 1,023 files (26,350,356,941 bytes before archive compression) are distributed
as 27 independent archives in the repository's `snapshot-2026-11-11` GitHub Release.
They include all server checkpoints, prediction arrays, prepared arrays, historical
experiment bundles and compatibility wheels. Search, diagnostic and smoke runs are
retained alongside selected runs; the experiment index identifies their roles.

`server_manifest.json` records every original file, its original relative path, byte size,
SHA-256 digest and storage location. Its `omissions` list covers only regenerable Python
bytecode, package metadata, runtime caches and lock files. Additional source and analysis
materials have separate provenance in `docs/BOOTSTRAP_PROVENANCE.json`.

Once downloaded, each archive can be independently extracted into the repository root.
For an automated download with archive and per-file verification, use Python 3.12+:

```bash
python scripts/restore_artifacts.py
python scripts/restore_artifacts.py --verify-only
```

To restore a subset, use e.g. `--part trefin-artifacts-01.tar.gz`. Consult the manifest
to find the part containing a particular checkpoint or prediction file. The restore
script downloads one archive at a time, verifies it, extracts it safely, then removes
only its own temporary archive. Allow at least 28 GB of free space for a full restore.

`release_assets.json` contains the exact GitHub download URLs and archive SHA-256 values.
File hashes are preserved without newline conversion through the root `.gitattributes`.
The existing scientific files were not retrained or rewritten for publication.

## Historical transport bundle

The original `server_bundle.tar.gz` is an incomplete historical transfer archive
(an existing gzip/tar truncation). It is retained byte-for-byte for provenance, but
is not a restoration source. Use the 27 verified release archives and the current
working-tree source instead. The other original historical bundles retain their
original contents and names.
