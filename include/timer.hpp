#pragma once
#include <cstddef>
#include <iostream>
#include <thread>

namespace myUtilities
{

    class Timer
    {
        bool         m_check{true};
        std::jthread m_thread;

      public:
        bool get_state() { return m_check; };
        // Input is assumed in seconds
        void start_timer(int n, bool verbose = false)
        {
            m_thread = std::jthread(
                [this, v = verbose](int n)
                {
                    if (v)
                        std::cout << "Thread : " << std::this_thread::get_id()
                                  << " is the one timing  " << n << " sec\n";
                    std::this_thread::sleep_for(std::chrono::seconds(n));
                    m_check = false;
                    std::cout << "Time elapsed \n";
                },
                n);
        }
    };

    class ChronoTimer
    {

      public:
        inline ChronoTimer(std::string label = "")
            : m_Label(std::move(label)), m_Start(std::chrono::high_resolution_clock::now())
        {
        }

        ~ChronoTimer() = default;

        void inline reset() { m_Start = std::chrono::high_resolution_clock::now(); }
        void inline get_time()
        {
            m_End   = std::chrono::high_resolution_clock::now();
            m_Delta = m_End - m_Start;
        };

        void inline print_time(size_t N_ops = 1)
        {
            get_time();
            std::cout << m_Label << " needed : " << m_Delta.count() / N_ops << "s\n";
        };

      private:
        std::string                                    m_Label;
        std::chrono::high_resolution_clock::time_point m_Start;
        std::chrono::high_resolution_clock::time_point m_End;
        std::chrono::duration<double>                  m_Delta;
    };

} // namespace myUtilities
