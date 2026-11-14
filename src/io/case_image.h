/**
 * @file case_image.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The case view's annotated images: SVG drawn from the view's blocks,
 *        with no renderer outside Mallard.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CASE_IMAGE_H
#define CASE_IMAGE_H

#include <string>
#include <vector>

#include "case_view.h"

namespace case_image {

/** @brief Most cells of the mesh whose edges a 2D image draws. */
constexpr size_t MESH_EDGE_CELLS = 10000;

/** @brief Text with &, <, > and " escaped, for XML text and attributes. */
std::string xml_text(const std::string & text);

/**
 * @brief The image of the case view's blocks (every rank's), as SVG in the
 *        documentation's light or dark palette. 2D: the plane, with the mesh's
 *        edges up to MESH_EDGE_CELLS cells; 3D: an isometric view over three
 *        orthographic ones (x-y, x-z, z-y), surfaces as their outlines, feature
 *        edges and silhouettes over translucent faces. Boundary zones are colored
 *        by the type of their condition, and every block with something to draw
 *        is labeled (the 3D isometric view) by its label, or its name, which also
 *        titles its group of paths (a tooltip in browsers); a legend,
 *        axes, a scale bar and the domain's extent with its units complete it.
 *        Filled regions show their boundary (case_view::Block::outline): exact
 *        where their shapes give it, else extracted below the cell size.
 *        The same blocks give the same text.
 */
std::string svg(const std::vector<case_view::Block> & blocks, bool dark, const std::string & title);

} // namespace case_image

#endif // CASE_IMAGE_H
