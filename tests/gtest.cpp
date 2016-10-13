#include "gtest/gtest.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int code = RUN_ALL_TESTS();
    return code;
}
