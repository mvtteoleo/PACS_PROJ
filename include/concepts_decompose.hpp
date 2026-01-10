#pragma once

#include <concepts>
#include <type_traits>

#include "communicator.hpp"

template <typename X>
concept CanBeUnpacked3 = (requires(X x) { std::tuple_size<X>::value == 3; }) || (requires(X x) {
                             { x[0] } -> std::convertible_to<int>;
                             { x[1] } -> std::convertible_to<int>;
                             { x[2] } -> std::convertible_to<int>;
                         });

template <typename L, typename T = double>
concept DecomposeConc = std::derived_from<L, Communicator<T>> && requires(L d) {
    { d.xStart() } -> CanBeUnpacked3;
    { d.xStartWGhosts() } -> CanBeUnpacked3;
    { d.xSize() } -> CanBeUnpacked3;
    { d.dimsWithGhosts() } -> CanBeUnpacked3;
};
