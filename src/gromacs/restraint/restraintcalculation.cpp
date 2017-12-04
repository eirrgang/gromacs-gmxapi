//
// Created by Eric Irrgang on 10/30/17.
//

#include "gromacs/pulling/pull.h"
#include "restraintcalculation-impl.h"
#include "restraintcalculation.h"

namespace gmx
{
namespace restraint
{
ICalculation::~ICalculation() = default;

Calculation::Calculation(double time,
                         const t_commrec &commRec,
                         const t_mdatoms &atoms,
                         const t_pbc &pbc,
                         real lambda,
                         const rvec *positions,
                         pull_t *puller,
                         rvec *forces,
                         tensor virial) :
    time_{time}
{
//    energy_ = pull_potential(puller, atoms, pbc, commRec, time, lambda, positions, forces, virial, &work_);
}

real Calculation::energy() const noexcept
{
    return energy_;
}

real Calculation::work() const noexcept
{
    return work_;
}

double Calculation::time() const noexcept
{
    return time_;
}


}
}

