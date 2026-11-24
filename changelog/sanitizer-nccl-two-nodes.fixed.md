- `tools/dev/gpu_runtime_tests.sh` passes across nodes: synccheck and initcheck leave out NCCL's kernels (their
  barrier and flag-buffer errors are NCCL's own, by design), and the suppression of NCCL's handled driver
  entry-point errors also matches the stack of a communicator's first init, under libc's `pthread_once` (#1077).
