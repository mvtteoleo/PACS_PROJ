#pragma once
#include "compiler_directives.hpp"
#include "expressions.hpp"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace numPDE{
     template <typename T, size_t N = DEF_DIM, bool IsConst = false>
    class ElementProxy : public Expr<ElementProxy<T, N, IsConst>>
    {
        using PointerType = std::conditional_t<IsConst, const T*, T*>;
        PointerType base;
        size_t      dim;

    public:
        template <typename  R>
            requires requires (R r){ r.begin(); r.size(); }
        ElementProxy(R&& r) : base(std::addressof(r[0])), dim(N)
        {
        // static_assert(!std::is_rvalue_reference_v<R&&>, "Cannot create ElementProxy from a temporary object!");
        }

        ElementProxy(PointerType ptr, size_t size) : base(ptr), dim(size)
        {
#ifdef PEDANTIC
            assert(size == N && "Proxy size mismatch with Array size");
#endif
        }

        // Element access
        T& operator[](size_t i) const
            requires(!IsConst)
        {
            return base[i];
        }
        const T& operator[](size_t i) const { return base[i]; }
        size_t   size() const { return dim; }

        // Assignment from expression (mutable only)
        template <typename E>
            requires(!IsConst)
        ElementProxy& operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
#ifdef PEDANTIC
            assert(ex.size() == dim && "Size mismatch in assignment");
#endif
            for (size_t i = 0; i < dim; ++i)
                base[i] = ex[i];
            return *this;
        }

        // Assignment from initializer list (mutable only)
        ElementProxy& operator=(std::initializer_list<T> values)
            requires(!IsConst)
        {
#ifdef PEDANTIC
            assert(values.size() == dim && "Size mismatch in init list");
#endif
            std::copy_n(values.begin(), dim, base);
            return *this;
        }

        // Assignment from span (mutable only)
        ElementProxy& operator=(std::span<const T> values)
            requires(!IsConst)
        {
#ifdef PEDANTIC
            assert(values.size() == dim && "Size mismatch in span assignment");
#endif
            std::copy_n(values.begin(), dim, base);
            return *this;
        }

        // Implicit conversions for legacy compatibility
        operator std::span<T>()
            requires(!IsConst)
        {
            return {base, dim};
        }
        operator std::span<const T>() const { return {base, dim}; }
    };


};

