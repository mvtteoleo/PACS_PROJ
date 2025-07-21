#pragma once

namespace numPDE
{

    template <typename T>
    Tensor<T>::Tensor(std::vector<size_t> sizes)
        : m_Sizes{sizes}, m_Rank{sizes.size()},
          m_N_element{std::accumulate(sizes.begin(), sizes.end(), size_t(1), std::multiplies{})}
    {
        m_Datas.resize(m_N_element);
        m_Slices_size.resize(m_Rank);
        m_Slices_size[m_Rank - 1] = 1;
        for (int i = m_Rank - 2; i >= 0; --i)
            m_Slices_size[i] = m_Slices_size[i + 1] * m_Sizes[i + 1];
    }

    template <typename T>
    template <typename... Ts>
        requires UnsignedInt<Ts...>
    Tensor<T>::Tensor(Ts... idxs) : Tensor(std::vector<size_t>{static_cast<size_t>(idxs)...}) { }

    template <typename T>
    template <typename... Ts>
        requires UnsignedInt<Ts...>
    T& Tensor<T>::operator()(Ts... idxs)
    {
        std::vector indices{static_cast<size_t>(idxs)...};
        return (*this)(indices);
    }

    template <typename T>
    template <typename Ts>
        requires UnsignedInt<Ts>
    T& Tensor<T>::operator()(std::vector<Ts> indices)
    {
        if (indices.size() != m_Rank) throw "Dimensions not matching";
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i] >= m_Sizes[i]) throw std::out_of_range("Index out of bounds");

        size_t index = std::transform_reduce(std::execution::par, m_Slices_size.begin(),
                                             m_Slices_size.end(), indices.begin(), size_t(0));

        return m_Datas.at(index);
    }

} // namespace numPDE
