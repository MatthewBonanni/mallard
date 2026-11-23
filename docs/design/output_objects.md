# Output objects

How the files a run writes become objects identified by their content, which
the GUI's [content addressing](gui.md#content-addressing) relies on. User docs:
[Complete files](../guide/output.md#complete-files).

## Write-once outputs

Issue [#1326](https://github.com/MatthewBonanni/mallard/issues/1326). A file
under its own name is written once and never changed: what an output record
hashes stays true, a copy is byte-identical or does not exist, and the GUI can
move outputs as objects while the run goes on.

1. **A VTKHDF series is a file per snapshot plus a derived index.** With
   `series = true` (the default, and always for boundary, surface, curve and
   points snapshots) each snapshot is `PREFIX_<n>.vtkhdf`, a complete
   single-step VTKHDF file written under `.partial` and renamed. Its mesh
   arrays (`Points`, `Offsets`, `Connectivity`, `Types`; for PolyData the
   points, cells and fixed geometry arrays) are HDF5 external links to
   `PREFIX_mesh.vtkhdf`, written once at the series' first snapshot, so the
   mesh is still stored once. A region whose cells move (`subset_update`,
   isosurfaces) embeds its geometry in each snapshot instead.
2. **`PREFIX.vtkhdf` stays the file users open.** It is an index of HDF5
   virtual datasets (VDS) over the snapshots' files, with VTKHDF's temporal
   layout (`Steps`), so ParaView 5.12+, h5py, PyVista and `mallard` read it
   as before without knowing. Arrays every step shares (a fixed mesh) map
   once at offset 0, as the single file stored them; the others concatenate
   the steps' arrays, contiguous mappings merged. The one-row values
   (`NumberOf*`, `FieldData/TIME`, `STEP`), `Steps` and `mallard_snapshot`
   are real datasets, and `mallard_series_files` names each step's file. The
   index is derived: rank 0 rebuilds it whole from the snapshots' files after
   each snapshot (through `.partial` and a rename), so it is never the only
   copy of anything and is not hashed. VDS and external links name files
   relative to the index's directory (HDF5 resolves them from the opening
   file's directory first), so the directory can move as a whole.
3. **Continuations and 0.10 series.** A restarted run reads the index (or,
   for a series written in one file by 0.10 and earlier, renames it
   `PREFIX_legacy.vtkhdf` and describes its steps) and keeps the steps up to
   the restart time; the mesh file is checked against the run's mesh rather
   than rewritten. An XDMF series likewise keeps its `_mesh.h5`.
4. **A rewind sets files aside instead of overwriting them.** At setup, before
   any writer continues, rank 0 reads the run log's output records and moves
   every file that one of this run's writers wrote at a step after the
   restart step to `.rewound/<run_id>/` in the file's own directory (a rename
   on the same file system, whatever the file's size), then the ranks
   synchronize. The `start` record lists the moves (`restart.superseded`),
   and `mallard info` follows them. Files of writers the run does not have
   (a sibling branch from the same restart) stay where they are; a file a
   record up to the restart step also names is kept.
5. **Probe HDF5 and VTKHDF files stay growing files.** A point or curve probe
   writes a row per sample, often every step: a file per sample, or an index
   rebuilt per sample, would cost the step loop and fill directories. They
   are live files, read remotely by row ranges (#1317); a rewind copies one
   to `.rewound/<run_id>/` before cutting it back.

### Cost

The 240³ box of [the hash measurements](#chunk-tree-hashes-cost) on 3 ranks
(8 threads each), a restart file and a `series = true` VTKHDF snapshot (RHO,
U, P, T) every 2 steps, `main` against write-once files, two runs each:

| Run | Bytes | Hashed | `output_sync` | `output_async` | Time stepping | Wall |
|---|---|---|---|---|---|---|
| `main` (one series file) | 6.212 GB | 1.9 s, 1.74 s (restarts only) | 1.31 s, 1.37 s | 830, 860 ms | 13.9 s, 13.7 s | 88.3 s, 88.4 s |
| write-once | 6.212 GB | 1.84 s, 1.76 s (every file) | 1.16 s, 1.16 s | 647, 647 ms | 13.7 s, 13.4 s | 86.6 s, 85.7 s |

The files take the same bytes (6 211 681 255 against 6 212 328 428: the mesh once, index included), the step loop
does not wait longer, and every snapshot now has its hash. ParaView's reader
(VTK 9.7) reads the index as the four steps, each step's `RHO` equal to its
snapshot file's, from another working directory.

Rejected:

- **`series = false` as the default.** Write-once too, but every snapshot
  repeats the mesh (about twice the bytes for a few fields) and ParaView
  needs the `.vtkhdf.series` index rather than one VTKHDF file.
- **Sealing the single file at the end of the run.** Mutable for the whole
  run (no object to copy or verify until it ends), lost on a killed run, and
  a multi-TB file rehashed whole at the end.
- **Appending, and hashing a growing prefix.** Every snapshot rewrites HDF5
  metadata blocks inside the file, so earlier bytes change: there is no
  prefix to hash.

## Chunk-tree hashes

Issue [#1329](https://github.com/MatthewBonanni/mallard/issues/1329).

1. **A file of one chunk keeps its SHA-256; a larger one has a chunk tree.**
   The SHA-256 of each 64 MiB chunk, in order, and `sha256_tree`, the SHA-256
   of their 32-byte digests one after the other. An empty file is one chunk.
   Every file of up to 64 MiB, which is most of them (VTU pieces, small
   snapshots, probe snapshots), keeps the hash every tool already checks; a
   reader tells the two apart by the key. One level of tree is enough: a
   200 GB file has 3200 chunks, 100 KB of digests in its record.
2. **Every rank hashes a share of every file all ranks write.** Rank r of P
   hashes chunks [n r / P, n (r + 1) / P) of a shared file (VTKHDF and XDMF
   snapshots and mesh files, restart files) in its background hash thread,
   reading the file back with `pread` once all ranks have closed it (a
   barrier before the job is queued; a restart file once rank 0 has renamed
   it). The pieces are gathered in rank order with the snapshot's record, so
   the record is the same on any rank count. Rank 0 alone reading every byte
   back is gone: it cost about 25 s for a 40 GB restart and fell behind runs
   that write often. Each rank's own files (VTU pieces) are hashed whole by
   that rank, as before.
3. **The chunk size is in the record.** `chunk_bytes` goes with every
   `sha256_tree`, so a later default would not invalidate earlier records;
   tests set a small chunk to make small files of many chunks.
4. **Chunks are also the unit of transfer.** Any chunk of a copy verifies on
   its own, so a transfer resumes, checks and runs in parallel by chunk.

### Cost {#chunk-tree-hashes-cost}

A 240³ cartesian box in 3D (13.8M cells), 6 steps, a restart file (553 MB)
and a `series = false` VTKHDF snapshot (2.0 GB) every 2 steps, 10.2 GB in all,
on a CPU node, Threads backend, the same input on `main` and with this change.
"Hashed" is the time the hash threads spent (on the slowest rank); the step
loop's `output_sync` and `output_async` are per step, slowest rank.

| Run | Ranks | Hashed | `output_sync` | `output_async` | Wall |
|---|---|---|---|---|---|
| `main` | 1 | 7.56 s | 542 ms | 0 | not timed |
| chunk trees | 1 | 7.62 s | 582 ms | 0 | 169 s |
| `main` | 3 | 8.1 s | 1.89 s | 1.41 s | 89.0 s |
| chunk trees | 3 | 3.25 s | 2.13 s | 1.57 s | 90.4 s |

On one rank the hash costs the same (the same bytes read by one thread); on
three it falls with the rank count, and the step loop does not wait for it in
either case: the hash runs in the background after the write.
