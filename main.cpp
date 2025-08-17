
int main (int argc, char *argv[]) {
    // Ideally here there should be something like:





    numPDE::Mesh meshDatas(file.dat);
    numPDE::NS_DNS simulation(meshDatas);
    while( simulation.t_current < simulation.final_t)
    {
        simulation.update_timestep();
        if(t%meshDatas.dumpTime<1e-8)
            simulation.dumpBinary;
        if(t%meshDatas.writeTime<1e-8)
            simulation.writeVTK;
    }

    return 0;
}
