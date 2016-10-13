#include "gtest/gtest.h"
#include <iostream>
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int code = RUN_ALL_TESTS();
    std::cout << code << std::endl;
    return code;
}
