"""Read a Gmsh mesh with the gmsh Python module (pip install gmsh) for plots:
nodes, the cells of the highest dimension and the faces of each physical
group of the dimension below (the boundary zones Mallard sees)."""
import numpy as np


def read_msh(path):
    """Returns (xyz (n, 3), cells: list of node-index arrays, zones: {name: list of node-index arrays})."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = coords.reshape(-1, 3)
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(tags.size)
        dim = gmsh.model.getDimension()

        def elements(d, tag=-1):
            out = []
            for etype, _, enodes in zip(*gmsh.model.mesh.getElements(d, tag)):
                props = gmsh.model.mesh.getElementProperties(etype)
                n_nodes, n_corners = props[3], props[5]
                conn = index[np.asarray(enodes, dtype=np.int64)].reshape(-1, n_nodes)[:, :n_corners]
                out.extend(conn)
            return out

        cells = elements(dim)
        zones = {}
        for d, ptag in gmsh.model.getPhysicalGroups(dim - 1):
            name = gmsh.model.getPhysicalName(d, ptag) or f"physical_{ptag}"
            faces = []
            for ent in gmsh.model.getEntitiesForPhysicalGroup(d, ptag):
                faces.extend(elements(d, ent))
            zones[name] = faces
    finally:
        gmsh.finalize()
    return xyz, cells, zones
