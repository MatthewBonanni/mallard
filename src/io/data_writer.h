/**
 * @file data_writer.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief DataWriter class declaration.
 * @version 0.1
 * @date 2024-01-01
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef DATA_WRITER_H
#define DATA_WRITER_H

#include <toml.hpp>

#include "data.h"
#include "mesh.h"

enum class DataFormat {
    VTU,
    TECPLOT
};

static const std::unordered_map<std::string, DataFormat> FORMAT_TYPES = {
    {"vtu", DataFormat::VTU},
    {"tecplot", DataFormat::TECPLOT}
};

static const std::unordered_map<DataFormat, std::string> FORMAT_NAMES = {
    {DataFormat::VTU, "vtu"},
    {DataFormat::TECPLOT, "tecplot"}
};

class DataWriter {
    public:
        /**
         * @brief Construct a new DataWriter object
         */
        DataWriter();

        /**
         * @brief Destroy the DataWriter object
         */
        ~DataWriter();

        /**
         * @brief Initialize the DataWriter.
         * @param input TOML input parameter table.
         * @param data Data objects.
         * @param mesh pointer to the mesh.
         */
        void init(const toml::value & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh);

        /**
         * @brief Write the data.
         * @param step Current time step.
         * @param force Force write.
         */
        void write(uint32_t step, bool force = false) const;
    protected:
        /**
         * @brief Write the data in VTU format.
         */
        void write_vtu(uint32_t step) const;

        /**
         * @brief Write the data in Tecplot format.
         */
        void write_tecplot(uint32_t step) const;

        std::string prefix;
        uint32_t interval;
        DataFormat format;
        std::vector<const Data *> data_ptrs;
        std::shared_ptr<Mesh> mesh;
    private:
};

#endif // DATA_WRITER_H