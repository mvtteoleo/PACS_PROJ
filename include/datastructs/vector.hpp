#pragma once
#include "compiler_directives.hpp"
#include "expressions.hpp"
#include <cassert>
#include <vector>
#include <span>
#include <algorithm>
#include <array> // Added missing header for std::array

namespace numPDE
{
    template <typename T>
    struct Vector : public std::vector<T>, public Expr<Vector<T>>
    {
      public:
        using value_type = T;
        
        // --- 1. RESOLVE AMBIGUITY ---
        // Explicitly use std::vector's implementation for these methods
        using std::vector<T>::operator[];
        using std::vector<T>::size;
        using std::vector<T>::begin;
        using std::vector<T>::end;
        using std::vector<T>::resize; // Needed if accessed via Vector
        using std::vector<T>::vector;
        

        Vector() = default;

        template <typename E>
        Vector(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            this->resize(ex.size()); // Use this-> or local using declaration

            for (size_t i = 0; i < ex.size(); ++i)
                (*this)[i] = ex[i];
        }

        template <typename E>
        void operator()(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            this->resize(ex.size());
            for (size_t i = 0; i < ex.size(); ++i)
                (*this)[i] = ex[i];
        }

        template <typename E>
        Vector& operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            this->resize(ex.size());
            for (size_t i = 0; i < ex.size(); ++i)
                (*this)[i] = ex[i];
            return (*this);
        }

        // Assignment from span
        Vector& operator=(std::span<T> span)
        {
            this->resize(span.size());
            std::copy(span.begin(), span.end(), this->begin());
            return *this;
        }

        Vector& operator=(std::span<const T> span)
        {
            this->resize(span.size());
            std::copy(span.begin(), span.end(), this->begin());
            return *this;
        }
    };

    template <typename T, size_t N = DEF_DIM>
    struct Array : public std::array<T, N>, public Expr<Array<T, N>>
    {
      public:
        using value_type = T;

        // --- RESOLVE AMBIGUITY FOR ARRAY ---
        using std::array<T, N>::operator[];
        using std::array<T, N>::size;
        using std::array<T, N>::begin;
        using std::array<T, N>::end;
        using std::array<T, N>::array;

        Array() = default;
        
        // Initialize std::array with fill logic if needed, or loop
        Array(const T v) 
        { 
            std::fill(this->begin(), this->end(), v); 
        }

        Array(std::initializer_list<T> l)
        {
            assert(l.size() <= N && "Value bigger than the size of the element");
            std::copy(l.begin(), l.end(), this->begin());
        }

        template <typename E>
        void operator()(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
#ifdef PEDANTIC
            assert("Size mismatch" && ex.size() == N);
#endif
            for (size_t i = 0; i < ex.size(); ++i)
                (*this)[i] = ex[i];
        }

        template <typename E>
        Array& operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
#ifdef PEDANTIC
            assert("Size mismatch" && ex.size() == N);
#endif
            for (size_t i = 0; i < ex.size(); ++i)
                (*this)[i] = ex[i];
            return (*this);
        }

        Array& operator=(std::span<T> span)
        {
            // std::array cannot resize, so we assume span fits or copy N
            std::copy_n(span.begin(), N, this->begin());
            return *this;
        }

        Array& operator=(std::span<const T> span)
        {
            std::copy_n(span.begin(), N, this->begin());
            return *this;
        }
    };
} // namespace numPDE
