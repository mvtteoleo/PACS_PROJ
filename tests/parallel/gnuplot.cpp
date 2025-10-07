#include "../../deps/gnuplot-iostream.h"
#include "../../header/MY_LIB.hpp"
#include <cmath> // for sin()
#include <utility>
#include <vector>

#define TEST 2

#if TEST == 0
int main(int argc, char* argv[])
{
    // Generate sine data

    NewDecomp<> dec(argc, argv);
    Gnuplot     gp;

    // Generate data
    std::vector<std::pair<double, double>> data;
    for (double x = 0; x <= 2 * M_PI; x += 0.01)
    {
        data.emplace_back(x, std::sin(x));
    }

    // Configure plot appearance
    gp << "set title 'Sine Curve' font ',14'\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'sin(x)'\n";
    gp << "set grid\n";
    gp << "set key top right\n";

    // Plot with styles: lw=line width, lc=line color, lt=line type, pt=point type
    gp << "plot '-' with lines lw 2 lc rgb 'blue' title 'sin(x)'\n";
    gp.send1d(data);

    return 0;
}

#elif TEST == 1
#include <tuple>
#include <vector>

int main()
{
    Gnuplot gp;

    std::vector<std::tuple<double, double, double>> points;
    for (int i = 0; i < 1000; i++)
    {
        double x = i / 10.0;
        double y = std::sin(x);
        double z = std::cos(x);
        points.emplace_back(x, y, z);
    }

    gp << "set title '3D Scatter'\n";
    gp << "splot '-' with points pt 7 ps 1.5 lc rgb 'blue' title 'data'\n";
    gp.send1d(points);
    gp << "pause -1\n"; // wait until user closes window

    std::cout << "Press Enter to exit...";
    std::cin.get(); // waits until user presses Enter

    return 0;
}

#elif TEST == 2
#include <cmath>
#include <iostream>

#include <cmath>
#include <iostream>
#include <tuple>

int main()
{
    Gnuplot gp;

    // Choose sphere parameters
    double r  = 1.0;
    double xc = 0.0, yc = 0.0, zc = 0.0;
    double h = 0.2;

    // Generate data points for a smooth sphere

    int                                             n_theta = 100, n_phi = 360;
    std::vector<std::tuple<double, double, double>> exact_circle(n_theta * n_phi);
    std::vector<std::tuple<double, double, double>> appr;

    for (int i = 0; i <= n_theta; ++i)
    {
        // FORCE CIRCLE
        constexpr double theta = M_PI / 2; // M_PI * i / n_theta;
        for (int j = 0; j <= n_phi; ++j)
        {
            double           phi = 2 * M_PI * j / n_phi;
            double           x   = xc + r * std::sin(theta) * std::cos(phi);
            double           y   = yc + r * std::sin(theta) * std::sin(phi);
            constexpr double z   = 0; // zc + r * std::cos(theta);
            double           x_h = h * static_cast<int>(x / h + 1);
            double           y_h = h * static_cast<int>(y / h + 1);
            exact_circle.emplace_back(x, y, z);
            appr.emplace_back(x_h, y_h, z);
        }
    }

    // --- Here's the important part ---
    gp << "set terminal x11 size 800,600\n"; // ✅ interactive terminal
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";
    gp << "set zlabel 'Z'\n";
    gp << "set view 00, 00\n";
    gp << "set grid\n";
    gp << "set title 'Interactive 3D Sphere'\n";
    // Plot both datasets
    gp << "splot '-' with lines lc rgb 'blue' lw 2 title 'Exact Circle', "
          "'-' with points pt 5 ps 1.2 lc rgb 'red' title 'Square Approx'\n";

    // Plot and hold window open
    // gp << "splot '-' with pm3d notitle\n";
    gp.send1d(exact_circle);
    gp.send1d(appr);

    std::cout << "Press Enter to close...\n";
    std::cin.get();
}

#endif
