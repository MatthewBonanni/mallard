/**
 * @file data.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Data class declaration.
 * @version 0.1
 * @date 2024-01-01
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef DATA_H
#define DATA_H

#include <string>

#include "common_typedef.h"

class Data {
    public:
        /**
         * @brief Construct a new Data object
         * @param name Name of data array.
         * @param ptr Pointer to data array.
         * @param stride Stride, i.e. size of state array.
         */
        Data(std::string name, double * ptr, int stride = 1) :
            m_name(name), ptr(ptr), stride(stride) {
            // Empty
        }

        /**
         * @brief Destroy the Data object
         */
        ~Data() {
            // Empty
        }

        /**
         * @brief Access element of data array.
         * @param i Index of element to access.
         * @return Reference to element.
         */
        double & operator[](int i) const {
            // Account for size of conservative state arrays
            return ptr[i * stride];
        }

        /**
         * @brief Get the name of the data array.
         * @return Name of data array.
         */
        std::string name() const {
            return m_name;
        }
    protected:
        std::string m_name;
        double * ptr;
        int stride;
    private:
};

#endif // DATA_H