//
// Created by nishith on 1/23/18.
//
#include <sstream>
#include <string>

#ifndef TRIANGLE_COUNT_SSTREAMCONVERT_H
#define TRIANGLE_COUNT_SSTREAMCONVERT_H

template <typename T> T convert_to (const std::string &str)
{
    std::istringstream ss(str);
    T num;
    ss >> num;
    return num;
}
#endif //TRIANGLE_COUNT_SSTREAMCONVERT_H
