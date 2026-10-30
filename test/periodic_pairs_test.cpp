/**
 * @file periodic_pairs_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The pairs of [[periodic]]: what a generated mesh gives (the
 *        translation or angle between its opposite sides) and checks.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <numbers>
#include <string>
#include <vector>

#include "input.h"
#include "mesh.h"
#include "test_fixtures.h"

namespace {

const std::string BOX = "[mesh]\ntype = \"cartesian\"\nLx = 2.0\nLy = 3.0\nLz = 4.0\n";
const std::string SECTOR = "[mesh]\ntype = \"annulus\"\nangle = 60.0\nLz = 0.5\n";

std::vector<Mesh::PeriodicPair> pairs_of(const std::string & text) { return Mesh::periodic_pairs(parse_toml(text)); }

std::string error_of(const std::string & text) {
    try {
        pairs_of(text);
    } catch (const InputError & e) {
        return e.what();
    }
    return "";
}

/** @brief A TOML vector of N_DIM components. */
std::string components(const std::string & x, const std::string & y, const std::string & z) {
    return "[" + x + ", " + y + (N_DIM == 3 ? ", " + z : "") + "]";
}

std::array<rtype, N_DIM> along(size_t d, rtype length) {
    std::array<rtype, N_DIM> t{};
    t[d] = length;
    return t;
}

} // namespace

TEST(PeriodicPairsTest, OppositeSidesOfAGeneratedBoxAreTheBoxLengthApart) {
    const auto pairs = pairs_of(BOX + periodic_sides(N_DIM == 3 ? "xyz" : "xy") + "[[periodic]]\nzones = [\"right\", \"left\"]\n");
    ASSERT_EQ(pairs.size(), size_t(N_DIM) + 1);
    EXPECT_EQ(pairs[0].translation, along(0, 2.0_r));
    EXPECT_EQ(pairs[1].translation, along(1, 3.0_r));
    if constexpr (N_DIM == 3) EXPECT_EQ(pairs[2].translation, along(2, 4.0_r));
    // Zone b is zone a shifted: right to left is the other way
    EXPECT_EQ(pairs.back().zone_a, "right");
    EXPECT_EQ(pairs.back().translation, along(0, -2.0_r));
    for (const auto & pair : pairs) EXPECT_EQ(pair.angle, 0.0_r);
}

TEST(PeriodicPairsTest, TheAnnulusSidesAreItsAngleApart) {
    const auto pairs = pairs_of(SECTOR + "[[periodic]]\nzones = [\"start\", \"end\"]\n" +
                                "[[periodic]]\nzones = [\"end\", \"start\"]\n" + periodic_sides(N_DIM == 3 ? "z" : ""));
    ASSERT_EQ(pairs.size(), N_DIM == 3 ? 3u : 2u);
    const rtype angle = 60.0_r * rtype(std::numbers::pi / 180.0);
    EXPECT_EQ(pairs[0].angle, angle);
    EXPECT_EQ(pairs[1].angle, -angle);
    EXPECT_EQ(pairs[0].center, (std::array<rtype, N_DIM>{}));
    if constexpr (N_DIM == 3) {
        EXPECT_EQ(pairs[2].angle, 0.0_r);
        EXPECT_EQ(pairs[2].translation, along(2, 0.5_r));
    }
}

TEST(PeriodicPairsTest, AnExplicitValueIsCheckedAgainstTheGeneratedMesh) {
    // Within rounding of the mesh's own value: taken from the mesh, exactly
    const auto close = pairs_of(BOX + "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = " +
                                components("2.0000000000001", "0.0", "0.0") + "\n");
    ASSERT_EQ(close.size(), 1u);
    EXPECT_EQ(close[0].translation, along(0, 2.0_r));
    const auto sector = pairs_of(SECTOR + "[[periodic]]\nzones = [\"start\", \"end\"]\nangle = 60.0\n");
    ASSERT_EQ(sector.size(), 1u);
    EXPECT_EQ(sector[0].angle, 60.0_r * rtype(std::numbers::pi / 180.0));

    const std::string shorter =
        error_of(BOX + "[[periodic]]\nzones = [\"left\", \"right\"]\ntranslation = " + components("1.0", "0.0", "0.0") + "\n");
    EXPECT_NE(shorter.find("periodic[0].translation"), std::string::npos) << shorter;
    EXPECT_NE(shorter.find("translation = " + components("2.0", "0.0", "0.0")), std::string::npos) << shorter;
    const std::string sideways =
        error_of(BOX + "[[periodic]]\nzones = [\"bottom\", \"top\"]\ntranslation = " + components("3.0", "0.0", "0.0") + "\n");
    EXPECT_NE(sideways.find("translation = " + components("0.0", "3.0", "0.0")), std::string::npos) << sideways;
    const std::string reversed =
        error_of(BOX + "[[periodic]]\nzones = [\"right\", \"left\"]\ntranslation = " + components("2.0", "0.0", "0.0") + "\n");
    EXPECT_NE(reversed.find("translation = " + components("-2.0", "0.0", "0.0")), std::string::npos) << reversed;
    const std::string rotated = error_of(BOX + "[[periodic]]\nzones = [\"left\", \"right\"]\nangle = 90.0\n");
    EXPECT_NE(rotated.find("periodic[0].angle"), std::string::npos) << rotated;
    const std::string narrower = error_of(SECTOR + "[[periodic]]\nzones = [\"start\", \"end\"]\nangle = 50.0\n");
    EXPECT_NE(narrower.find("angle = 60.0"), std::string::npos) << narrower;
    const std::string shifted = error_of(SECTOR + "[[periodic]]\nzones = [\"start\", \"end\"]\ntranslation = " +
                                         components("1.0", "0.0", "0.0") + "\n");
    EXPECT_NE(shifted.find("angle = 60.0"), std::string::npos) << shifted;
}

TEST(PeriodicPairsTest, OtherPairsNeedATranslationOrAnAngle) {
    for (const std::string & text :
         {std::string("[mesh]\ntype = \"file\"\n[[periodic]]\nzones = [\"inlet\", \"outlet\"]\n"),
          BOX + "[[periodic]]\nzones = [\"left\", \"top\"]\n"}) {
        const std::string error = error_of(text);
        EXPECT_NE(error.find("neither a translation nor an angle"), std::string::npos) << error;
    }
    EXPECT_NE(error_of(BOX + "[[periodic]]\nzones = [\"left\"]\n").find("lists 1 zones"), std::string::npos);
    EXPECT_NE(error_of(BOX + "[[periodic]]\nzones = [\"left\", \"right\"]\nangle = 10.0\ntranslation = " +
                       components("2.0", "0.0", "0.0") + "\n")
                  .find("both a translation and an angle"),
              std::string::npos);
    // Explicit pairs other than opposite sides are the mesh's to check (by matching their nodes)
    const auto skewed = pairs_of(BOX + "[[periodic]]\nzones = [\"left\", \"top\"]\ntranslation = " +
                                 components("1.0", "1.0", "0.0") + "\n");
    ASSERT_EQ(skewed.size(), 1u);
    EXPECT_EQ(skewed[0].translation[1], 1.0_r);
}
