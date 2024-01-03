/**
 * @file data_writer.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief DataWriter class implementation.
 * @version 0.1
 * @date 2024-01-01
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "data_writer.h"

#include <string>
#include <optional>

DataWriter::DataWriter() {
    // Empty
}

DataWriter::~DataWriter() {
    // Empty
}

void DataWriter::init(const toml::table & input,
                      std::vector<Data> & data) {
    std::optional<std::string> prefix = input["prefix"].value<std::string>();
    std::optional<std::string> format_str = input["format"].value<std::string>();
    std::optional<int> interval = input["interval"].value<int>();
    auto variables = input["variables"];
    const toml::array* arr = variables.as_array();

    if (!prefix.has_value()) {
        throw std::runtime_error("DataWriter: prefix not specified.");
    }
    if (!format_str.has_value()) {
        throw std::runtime_error("DataWriter: format not specified.");
    }
    if (!interval.has_value()) {
        throw std::runtime_error("DataWriter: interval not specified.");
    }
    if (!variables) {
        throw std::runtime_error("DataWriter: variables not specified.");
    } else if (arr->size() == 0) {
        throw std::runtime_error("DataWriter: variables must be a non-empty array.");
    }

    this->prefix = prefix.value();
    this->interval = interval.value();

    DataFormat format;
    typename std::unordered_map<std::string, DataFormat>::const_iterator it = FORMAT_TYPES.find(*format_str);
    if (it == FORMAT_TYPES.end()) {
        throw std::runtime_error("DataWriter: Unknown format type: " + *format_str + ".");
    } else {
        format = it->second;
    }

    for (const auto & var : *arr) {
        std::optional<std::string> var_str = var.value<std::string>();
        if (!var_str.has_value()) {
            throw std::runtime_error("DataWriter: variable must be a string.");
        }

        bool found = false;
        for (const auto & data_var : data) {
            if (data_var.name() == var_str.value()) {
                data_ptrs.push_back(&data_var);
                found = true;
                break;
            }
        }
    }
}

void DataWriter::write(int step, bool force) const {
    bool write_now = (step % interval == 0) || force;
    if (!write_now) {
        return;
    }

    if (format == DataFormat::VTU) {
        write_vtu(step);
    } else if (format == DataFormat::TECPLOT) {
        write_tecplot(step);
    } else {
        throw std::runtime_error("DataWriter::write not implemented.");
    }
}

void DataWriter::write_vtu(int step) const {
    std::string filename = prefix + "_" + std::to_string(step) + ".vtu";

    throw std::runtime_error("DataWriter::write_vtu not implemented.");
}

void DataWriter::write_tecplot(int step) const {
    std::string filename = prefix + "_" + std::to_string(step) + ".dat";

    throw std::runtime_error("DataWriter::write_tecplot not implemented.");
}