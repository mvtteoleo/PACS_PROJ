Repo fatta con l'idea di sviluppare il codice per la tesi

nix develop per attivare la shell con tutto.
tests in ```./tests/```
```make <nome_test(SENZA.cpp)>``` per generare l'eseguibile

Milestones: 
  - Scalar and Vector Fields
  - Working math operators
  - Working dumps (binary and VTK format)
  - Working time step (Serial)                          
  - Implementation of the pressure correction (Serial and then parallel)
  - Full solver (And validation)
  - Add the porosity part to the problem (IBM)


Some more notes:
    - Curiously recurring templates instead of inheritance from virtual
    - Policy based designed instead of a solver class that inherits
    - Expression templates
    - Static reflections and compile time dispatch
