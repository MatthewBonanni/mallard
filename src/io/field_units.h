/**
 * @file field_units.h
 * @brief Units and descriptions of the fields Mallard writes: the one table
 *        that every output format takes its units metadata from.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FIELD_UNITS_H
#define FIELD_UNITS_H

#include <optional>
#include <string>
#include <vector>

#include "units.h"

/** @brief What an output field holds: its dimension (the field is in SI units) and a description. */
struct FieldDescription {
    Dimension dimension;
    std::string long_name;
    std::string standard_name;  // the CF standard name, empty where none fits

    /** @brief The field's units in UDUNITS syntax ("kg m-3", "1"). */
    std::string units() const { return dimension.udunits(); }
};

namespace field_units {

/**
 * @brief The description of an output field by name: a named field (RHO,
 *        U_X, U, ...), a per-species field (Y_<s>, OMEGA_<s>, ...), or one
 *        derived from others (SOURCE_<A>, MEAN_<A>, COV_<A>_<B> with the
 *        product of their units, PHASE_<bin>_<A>); nullopt if the table has
 *        no such field.
 */
std::optional<FieldDescription> find(const std::string & name);

/**
 * @brief find, for a field that must be in the table.
 * @throws std::logic_error naming the field if it is not
 */
FieldDescription describe(const std::string & name);

/** @brief The comment line under a CSV header: "# units: " and each column's units, comma-separated. */
std::string csv_line(const std::vector<Dimension> & columns);

/** @brief text with &, <, > and " escaped for an XML attribute value. */
std::string xml_escape(const std::string & text);

/**
 * @brief The VTK XML field-data arrays (one String array each) that record the
 *        units and descriptions of the named arrays: MALLARD_FIELD_NAMES,
 *        MALLARD_FIELD_UNITS and MALLARD_FIELD_LONG_NAMES, entry i of each for
 *        names[i]; every line starts with indent.
 */
std::string vtk_field_data(const std::vector<std::string> & names, const std::vector<FieldDescription> & fields,
                           const std::string & indent);

} // namespace field_units

#endif // FIELD_UNITS_H
