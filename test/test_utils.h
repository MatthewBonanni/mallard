/**
 * @file test_utils.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Utilities for testing.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <gtest/gtest.h>

#ifdef Mallard_USE_DOUBLE
    #define EXPECT_RTYPE_EQ EXPECT_DOUBLE_EQ
#else
    #define EXPECT_RTYPE_EQ EXPECT_FLOAT_EQ
#endif

#endif // TEST_UTILS_H