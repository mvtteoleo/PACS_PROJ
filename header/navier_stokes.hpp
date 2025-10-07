#pragma once

class NSproblem
{
  public:
    NSproblem();
    NSproblem(NSproblem&&)                 = default;
    NSproblem(const NSproblem&)            = default;
    NSproblem& operator=(NSproblem&&)      = default;
    NSproblem& operator=(const NSproblem&) = default;
    ~NSproblem();

  private:
};
