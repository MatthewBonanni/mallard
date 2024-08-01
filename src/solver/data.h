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

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

class Data {
    public:
        /**
         * @brief Construct a new Data object
         * @param name Name of data array.
         * @param view Pointer to view of data array.
         */
        Data(std::string name, Kokkos::View<rtype *, Kokkos::LayoutStride>::HostMirror view) :
            m_name(name), m_view(view) {
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
        rtype & operator[](size_t i) const {
            // Account for size of conservative state arrays
            return m_view(i);
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
        Kokkos::View<rtype *, Kokkos::LayoutStride>::HostMirror m_view;
    private:
};

#endif // DATA_H