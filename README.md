Repo fatta con l'idea di sviluppare il codice per la tesi

nix develop per attivare la shell con tutto.
tests in ```./tests/```
```make <nome_test(SENZA.cpp)>``` per generare l'eseguibile

Milestones: 
   Scalar and Vector Fields
   Working math operators
   Working dumps (binary and VTK format)
   Working time step (Serial)                          
   Implementation of the pressure correction (Serial and then parallel)
  - Full solver (And validation)
   Add the porosity part to the problem (IBM)


Some more notes:
    - Curiously recurring templates instead of inheritance from virtual
    - Policy based designed instead of a solver class that inherits
    - Expression templates
    - Static reflections and compile time dispatch


TODO 6 Settembre:
    - Investigare bene DCT e DST scale factor
    - Implementa il passaggio di informazioni, sfrutta 2Decomp gia che lo hai lì.
    - Iniziare a ragionare una classe solvePoisson e in generale una classe solve 
        visto che le cose ormai iniziano a funzionare
