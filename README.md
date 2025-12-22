Repo fatta con l'idea di sviluppare il codice per la tesi

nix develop per attivare la shell con tutto.
tests in ```./tests/```
```make <nome_test(SENZA.cpp)>``` per generare l'eseguibile

Milestones: 
  ✓ Scalar and Vector Fields
  ✓ Working math operators
  ✓ Working dumps (binary and VTK format)
  ✓ time step (Serial)                          
  ✓ of the pressure correction (Serial and then parallel)
  - Full solver (And validation)
  - Test the Poisson solvers for this part:

TODO:

  - Update pos in the outermost loop because efficiency (and correct the wrong updates)
  - Make ErrorHandler struct to avoid the MPI_Reduce etc and make it cleaner.
  - Check the L2 norm for time integration ( L2 = √( dt * ∑ err(t)^2) )!!!

Some more notes:

  - Curiously recurring templates instead of inheritance from virtual
  - Policy based designed instead of a solver class that inherits
  - Expression templates
  - Static reflections and compile time dispatch


# TODO for the report:

  - Full NS solver templated on the Poisson solver
  - Write VTK also for the pressure without intermediate Tensor
  - RK class to handle the time integration (If have time)
  - Scalability test
  - Parse from text



