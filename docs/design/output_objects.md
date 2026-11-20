# Output objects

How the files a run writes become objects identified by their content, which
the GUI's [content addressing](gui.md#content-addressing) relies on. User docs:
[Complete files](../guide/output.md#complete-files).

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

### Cost

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
