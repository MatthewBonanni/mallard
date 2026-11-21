/**
 * @file json_test.cpp
 * @brief Tests for the strict JSON parser in common/json.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "json.h"

TEST(JsonTest, ParsesNumbersInTheRfc8259Grammar) {
    EXPECT_EQ(json::parse("0").number, 0.0);
    EXPECT_EQ(json::parse("-0.5").number, -0.5);
    EXPECT_EQ(json::parse("12.25e2").number, 1225.0);
    EXPECT_EQ(json::parse("1E-2").number, 0.01);
    EXPECT_EQ(json::parse("2e+3").number, 2000.0);
    const json::Value array = json::parse("[1,-2.5e1]");
    ASSERT_EQ(array.array.size(), 2u);
    EXPECT_EQ(array.array[1].number, -25.0);
}

TEST(JsonTest, RejectsMalformedNumbers) {
    for (const std::string text : {"1-2", "1.2.3", "1e", "1e+", "1.", "-", "-.5", ".5", "01", "1+2", "0x10", "[1e]",
                                   "{\"a\":1.2.3}"}) {
        EXPECT_THROW(json::parse(text), std::runtime_error) << text;
    }
}
