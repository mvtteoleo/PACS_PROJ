#pragma once
#include <iostream>
#include <vector>

// converts a variable name to a string
#define OUT_NAME(var) #var

#include <ranges>

/*
 * Concept to check the "iterability" of n element.
 */
template <typename T>
concept ElementIterable = requires(std::ranges::range_value_t<T> x) {
    x.begin();
    x.end();
};

/*
 * @brief : Allow to print ranges on a more reasonable manner.
 */
template <ElementIterable Range>
std::ostream& operator<<(std::ostream& os, const Range&& vec)
{
    std::cout << " : ( ";
    for (const auto& i : vec)
        std::cout << i << std::endl;
    std::cout << ") " << std::endl;
    return os;
}
