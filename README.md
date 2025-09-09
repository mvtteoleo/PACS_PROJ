Repo fatta con l'idea di sviluppare il codice per la tesi

nix develop per attivare la shell con tutto.
tests in ```./tests/```
```make <nome_test(SENZA.cpp)>``` per generare l'eseguibile

Milestones: 
   Scalar and Vector Fields
   Working math operators
   Working dumps (binary and VTK format)
   Working time step (Serial)                          
  👌Implementation of the pressure correction (Serial and then parallel)
  - Full solver (And validation)
   Add the porosity part to the problem (IBM)
    - Handle the 2/3 domains
    - Handle the different algorithms to make them sincronized
        - Here the problem is stiffer I can maybe use smaller time steps or a more refined RK scheme
    - Test the Poisson solvers for this part:
        - Iterative fft based 
            - Bad for memory,
            - But should be faster
        - Dumb Jacobi iterations 
            - Good for memory + matches well the mom eq solver
            - Not efficent, may lead to scaling problems


Some more notes:
    - Curiously recurring templates instead of inheritance from virtual
    - Policy based designed instead of a solver class that inherits
    - Expression templates
    - Static reflections and compile time dispatch


TODO 7 Settembre:
    ⏳ Implementa il passaggio di informazioni, sfrutta 2Decomp gia che lo hai lì.
    ⌛ Iniziare a ragionare una classe solvePoisson e in generale una classe solve 
        visto che le cose ormai iniziano a funzionare
