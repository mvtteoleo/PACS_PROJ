Repo fatta con l'idea di sviluppare il codice per la tesi

Nix develop per attivare la shell con tutto.
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



# To produce the container

1. Build the binaries
nix build .#container

1. Unzip the Nix output into a clean tar file
gunzip -c result > image.tar

1. Build the Apptainer image from the uncompressed tar
apptainer build pacs-env.sif docker-archive://image.tar

1. Clean up the temporary tar
rm image.tar
