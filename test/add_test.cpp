#include <gtest/gtest.h>
#include "../src/add.h"

TEST(AddTest, PositiveValues) {
    EXPECT_EQ(add(2, 3), 5);
}

TEST(AddTest, NegativeValues) {
    EXPECT_EQ(add(-2, -3), -5);
}

TEST(AddTest, MixedValues) {
    EXPECT_EQ(add(5, -3), 2);
}