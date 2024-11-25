/**
 * @file basis.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Basis functions class declaration.
 * @version 0.1
 * @date 2024-11-20
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef BASIS_H
#define BASIS_H

#include <unordered_map>

enum class BasisType {
    Legendre,
};

static const std::unordered_map<std::string, BasisType> BASIS_TYPES = {
    {"Legendre", BasisType::Legendre},
};

static const std::unordered_map<BasisType, std::string> BASIS_NAMES = {
    {BasisType::Legendre, "Legendre"},
};

class Basis {
    public:
        /**
         * @brief Construct a new Basis Function object
         */
        Basis();

        /**
         * @brief Destroy the Basis Function object
         */
        ~Basis();
};

class Legendre : public Basis {
    public:
        /**
         * @brief Construct a new Shifted Legendre object
         */
        Legendre();

        /**
         * @brief Destroy the Shifted Legendre object
         */
        ~Legendre();
};

#endif // BASIS_H