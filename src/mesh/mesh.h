/**
 * @file mesh.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh class declaration.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef MESH_H
#define MESH_H

#include <array>
#include <vector>

class Mesh {
    public:
        /**
         * @brief Construct a new Mesh object
         */
        Mesh();

        /**
         * @brief Destroy the Mesh object
         */
        ~Mesh();

        /**
         * @brief Get the number of cells.
         * @return Number of cells.
         */
        int n_cells() const;

        /**
         * @brief Get the number of nodes.
         * @return Number of nodes.
         */
        int n_nodes() const;

        /**
         * @brief Get the number of faces.
         * @return Number of faces.
         */
        int n_faces() const;

        /**
         * @brief Get the coordinates of a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Coordinates of the cell.
         */
        std::array<double, 2> cell_coords(int i_cell) const;

        /**
         * @brief Get the coordinates of a node.
         * 
         * @param i_node Index of the node.
         * @return Coordinates of the node.
         */
        std::array<double, 2> node_coords(int i_node) const;

        /**
         * @brief Get the volume of a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Volume of the cell.
         */
        double cell_volume(int i_cell) const;

        /**
         * @brief Get the area of a face.
         * 
         * @param i_face Index of the face.
         * @return Area of the face.
         */
        double face_area(int i_face) const;

        /**
         * @brief Get the normal of a face.
         * 
         * @param i_face Index of the face.
         * @return Unit vector normal to the face.
         */
        std::array<double, 2> face_normal(int i_face) const;

        /**
         * @brief Get the nodes comprising a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Array of node ids comprising the cell.
         */
        std::array<int, 4> nodes_of_cell(int i_cell) const;

        /**
         * @brief Get the faces comprising a cell.
         * 
         * @param i_cell Index of the cell.
         * @return Array of face ids comprising the cell.
         */
        std::array<int, 4> faces_of_cell(int i_cell) const;

        /**
         * @brief Get the cells bounding a face.
         * 
         * @param i_face Index of the face.
         * @return Array of cell ids bounding the face.
         */
        std::array<int, 2> cells_of_face(int i_face) const;

        /**
         * @brief Get the nodes comprising a face.
         * 
         * @param i_face Index of the face.
         * @return Array of node ids comprising the face.
         */
        std::array<int, 2> nodes_of_face(int i_face) const;

        /**
         * @brief Compute cell centroids.
         */
        void compute_cell_centroids();

        /**
         * @brief Compute cell volumes.
         */
        void compute_cell_volumes();

        /**
         * @brief Compute face areas.
         */
        void compute_face_areas();

        /**
         * @brief Compute face normals.
         */
        void compute_face_normals();

        /**
         * @brief Initialize the supersonic wedge mesh.
         * 
         * @param nx Number of cells in the x-direction.
         * @param ny Number of cells in the y-direction.
         * @param Lx Length of the domain in the x-direction.
         * @param Ly Length of the domain in the y-direction.
         */
        void init_wedge(int nx, int ny, double Lx, double Ly);
    protected:
    private:
        int nx, ny;
        std::vector<std::array<double, 2>> m_node_coords;
        std::vector<std::array<double, 2>> m_cell_coords;
        std::vector<double> m_cell_volume;
        std::vector<double> m_face_area;
        std::vector<std::array<double, 2>> m_face_normals;
        std::vector<std::array<int, 4>> m_nodes_of_cell;
        std::vector<std::array<int, 4>> m_faces_of_cell;
        std::vector<std::array<int, 2>> m_cells_of_face;
        std::vector<std::array<int, 2>> m_nodes_of_face;
};

#endif // MESH_H