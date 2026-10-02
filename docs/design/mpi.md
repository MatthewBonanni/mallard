# Design note: distributed memory (MPI), issue #2

Status: not implemented; MPI was not available on the development machine.
This note records how the current code would extend to MPI with the least
disruption.

## Partitioning

- Partition cells once at startup on rank 0, either recursive coordinate
  bisection on cell centroids (no dependencies) or METIS/ParMETIS when
  available. Each rank owns a contiguous range of cells.
- Each rank builds its local mesh with `Mesh::init_from_connectivity` from its
  owned cells plus a halo of ghost cells. Local indices order owned cells
  first, then ghosts, so kernels that update cells loop over owned cells only.

## Halo depth

The halo must contain every cell any owned cell's reconstruction reads:

| Reconstruction | Halo |
|---|---|
| FO | 1 layer (face neighbors) |
| MUSCL | 2 layers (gradients and limiter of the neighbor across each face) |
| TENO | the union of the central and sector stencils, about 3 to 4 vertex-neighbor layers for order 5; compute it exactly from the stencil lists after the serial-style precomputation on the owned plus halo cells |

Viscous fluxes need 2 layers (vertex-neighbor gradients).

## Communication per right-hand side

1. Exchange the conservative state of halo cells (one message per neighbor rank, packed with Kokkos parallel_for into contiguous buffers; GPU-aware MPI where available).
2. Compute W, gradients and reconstruction on owned cells plus the halo layers that reconstruction needs.
3. Faces between an owned cell and a halo cell are computed by the rank owning the face's cell 0 or, simpler, by both ranks with only the owned side's residual kept; the second avoids a reverse exchange of fluxes and costs a few redundant face fluxes.
4. Global reductions: time step (min), conservation diagnostics (sum), average-pressure outlets (sum of p A and A).

## Boundaries

- Transmissive image faces and TENO mirror images are found during
  precomputation; with a halo of sufficient depth they resolve to local cells.
- Dirichlet face states are evaluated per rank for its own boundary faces.

## Output

- VTU: each rank writes a piece; rank 0 writes a `.pvtu` index and the `.pvd`.
- Restart: one file per rank, or gather to rank 0 for small cases.

## Testing

Run the existing solver tests on 1, 2 and 4 ranks and require identical
results to round-off (the partition only changes summation order of
`atomic_add` contributions).
