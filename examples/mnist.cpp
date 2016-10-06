#include "mnist.h"

#include <iostream>
#include <fstream>

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

Eigen::MatrixXf mnist::readMNISTImages(const std::string &fname) {
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);

    int magic;
    in.read((char *) &magic, sizeof(magic));
    magic = reverseInt(magic);
    if (magic != 2051) {
        throw std::string("bad magic");
    }
    int countOfImages;
    in.read((char *) &countOfImages, sizeof(countOfImages));
    countOfImages = reverseInt(countOfImages);

    int rows;
    in.read((char *) &rows, sizeof(rows));
    rows = reverseInt(rows);

    int cols;
    in.read((char *) &cols, sizeof(cols));
    cols = reverseInt(cols);

    printf("images %d rows %d cols %d\n", countOfImages, rows, cols);

    Eigen::MatrixXf res;
    res.resize(countOfImages, rows * cols);
    for (int i = 0; i < countOfImages; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char temp = 0;
            in.read((char *) &temp, sizeof(temp));
            res(i, j) = temp;
//            std::cout << i << ":" << j << ":" << res(i, j) << " ";
        }
    }
    in.close();
    return res;
}

Eigen::VectorXi mnist::readMNISTLabels(const std::string &fname) {
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);
    int magic;
    in.read((char *) &magic, sizeof(magic));
    magic = reverseInt(magic);

    int countOfLabels;
    in.read((char *) &countOfLabels, sizeof(countOfLabels));
    countOfLabels = reverseInt(countOfLabels);

    printf("magic %d %d\n", magic, countOfLabels);

    Eigen::VectorXi res = Eigen::VectorXi::Zero(countOfLabels);
    unsigned char tmp;
    for(int i = 0; i < countOfLabels; ++i) {
        in.read((char *) &tmp, sizeof(tmp));
        res[i] = tmp;
    }

    in.close();
    return res;
}
