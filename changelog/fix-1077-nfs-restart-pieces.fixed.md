- Asynchronous and compressed restart files written by ranks on several nodes to a shared NFS volume no longer
  come out with zeroed pages: rank 0 now gives the file its final size before the ranks write their pieces, so
  no client zero-fills a page past its stale end of file and writes it back over another rank's bytes.
