#include "mnist.h"

/*
 * ml::DataSet readMNISTData(std::string fname)
{
    ml::DataSet res;
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);

    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);
    if(magic != 2051){
        throw std::string("bad magic");
    }
    int countOfImages;
    in.read((char*)&countOfImages, sizeof(countOfImages));
    countOfImages = reverseInt(countOfImages);

    int rows;
    in.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);

    int cols;
    in.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    printf("images %d rows %d cols %d\n", countOfImages, rows, cols);

    res.resize(countOfImages, NULL);
    for(int i = 0; i < countOfImages; ++i){
        ml::FeatureVec *feats = new ml::FeatureVec(rows*cols, 0);
        for(int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                unsigned char temp=0;
                in.read((char*)&temp,sizeof(temp));
                (*feats)[r*rows + c] = temp;
                //printf("%f\n", (float)temp);
            }
        }
        res[i] = feats;
    }

    in.close();
    return res;
}

std::vector<int> readMNISTLabels(std::string fname)
{
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);
    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);

    int countOfLabels;
    in.read((char*)&countOfLabels, sizeof(countOfLabels));
    countOfLabels = reverseInt(countOfLabels);

    printf("magic %d %d\n", magic, countOfLabels);

    std::vector<int> res(countOfLabels, 0);
    unsigned char tmp;
    int counter = 0;
    while(in.good()){
        in.read((char*)&tmp, sizeof(tmp));
        res[counter] = tmp;
        counter++;
    }

    in.close();
    return res;
}
*/