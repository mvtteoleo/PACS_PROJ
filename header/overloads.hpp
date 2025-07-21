#pragma once
#include <iostream>
#include <vector>

// converts a variable name to a string
#define OUT_NAME(var) #var

// Print vectors pyhton-like
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> vec)
{
    std::cout << OUT_NAME(vec) << " : [ ";
    for (T i : vec)
        std::cout << i << std::endl;
    std::cout << "] " << std::endl;
    return os;
}
