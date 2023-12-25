/**
 * @file common_typedef.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Typedefs for common types.
 * @version 0.1
 * @date 2023-12-25
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_TYPEDEF_H
#define COMMON_TYPEDEF_H

#include <array>
#include <vector>

#define State std::array<double, 4>
#define StateVector std::vector<State>

#endif // COMMON_TYPEDEF_H