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

#include <toml++/toml.hpp>

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
        void init(const toml::table & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh);

        /**
         * @brief Write the data.
         * @param step Current time step.
         * @param force Force write.
         */
        void write(int step, bool force = false) const;
    protected:
        /**
         * @brief Write the data in VTU format.
         */
        void write_vtu(int step) const;

        /**
         * @brief Write the data in Tecplot format.
         */
        void write_tecplot(int step) const;

        std::string prefix;
        int interval;
        DataFormat format;
        std::vector<const Data *> data_ptrs;
        std::shared_ptr<Mesh> mesh;
    private:
};

#endif // DATA_WRITER_H