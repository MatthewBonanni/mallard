# Mallard for CPUs: OpenMP and MPI (with dKaMinPar), HDF5, CGNS and zlib; 2D (mallard2d) and 3D (mallard3d), both run as mallard.
#
#   docker build -f docker/Dockerfile -t mallard .
#
# from a checkout with the Kokkos and toml11 submodules.
FROM ubuntu:26.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libopenmpi-dev openmpi-bin libhdf5-openmpi-dev zlib1g-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
ARG JOBS
RUN JOBS=${JOBS} /src/docker/build.sh /src /opt/mallard -DKokkos_ENABLE_OPENMP=ON -DMallard_ENABLE_KAMINPAR=ON -DMallard_PYTHON_INSTALL_DIR=/usr/lib/python3/dist-packages

FROM ubuntu:26.04
LABEL org.opencontainers.image.title="Mallard" \
      org.opencontainers.image.description="High-order finite volume solver for compressible reacting flows (CPU: OpenMP + MPI)" \
      org.opencontainers.image.source="https://github.com/MatthewBonanni/mallard" \
      org.opencontainers.image.licenses="Apache-2.0"
# OpenSSH for mpirun across the pods of a Kubernetes MPIJob; sshd makes its host keys where it runs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 openmpi-bin libopenmpi40 libhdf5-openmpi-310 libtbb12 \
        python3 python3-numpy python3-scipy python3-matplotlib python3-h5py openssh-client openssh-server \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*
# Mallard's executables only, not the bundled Kokkos's tools
COPY --from=build /opt/mallard/bin/mallard* /opt/mallard/bin/
COPY --from=build /opt/mallard/share /opt/mallard/share
# The Python package mallard in the image's Python, and the reader library it loads
COPY --from=build /opt/mallard/lib/libmallard* /opt/mallard/lib/
COPY --from=build /usr/lib/python3/dist-packages/mallard /usr/lib/python3/dist-packages/mallard
# Tab completion of mallard in interactive shells
RUN echo '[ -r /opt/mallard/share/bash-completion/completions/mallard ] && . /opt/mallard/share/bash-completion/completions/mallard' \
        >> /etc/bash.bashrc
ENV PATH=/opt/mallard/bin:$PATH \
    MALLARD_SHARE=/opt/mallard/share/mallard \
    OMP_PROC_BIND=spread \
    OMP_PLACES=threads \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /work
