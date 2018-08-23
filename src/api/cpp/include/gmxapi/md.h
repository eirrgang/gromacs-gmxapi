/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef GMXAPI_MD_H
#define GMXAPI_MD_H
/*! \file
 * \brief Declare base classes and API for MD simulation engines.
 *
 * This header allows interaction with gmxapi for basic MD simulation functionality in client code without
 * additional dependencies.
 *
 * Helper functions, standard concrete classes, and implementation interfaces are in gmxapi/md/
 * \ingroup gmxapi_md
 */
/*! \dir md
 * \brief Additional declarations for API implementation and extension.
 *
 * This directory contains headers that require gromacsfwd.h and that declare
 * objects that have stronger dependencies on GROMACS to fully define or extend.
 * \ingroup gmxapi_md
 */

/*! \defgroup gmxapi_md Molecular Dynamics
 * \brief API access to Molecular Mechanics and Molecular Dynamics calculation in GROMACS
 *
 * At a minimum, the client code must specify the input source for a MD simulation. Helper functions allow setting up
 * inputs from a standard GROMACS run input `.tpr` file. A System object serves as a container for a molecular system
 * and associated computational work.
 *
 * \ingroup gmxapi
 */
#include <memory>
#include <string>
#include <vector>

#include "gmxapi/exceptions.h"

namespace gmxapi
{

/*! \addtogroup gmxapi_md

 # Extending MD with a custom restraint potential

   Below, we show code that extends the GROMACS library, code that interfaces with this
   API, and client code that calls GROMACS with the custom code.
   Refer to the sample plugin repository for a more complete example.

   \todo Move this to an example source code file that we can compile and test.

 ## Example

   In client code, extend the GROMACS library by implementing a new restraint
   potential (see library documentation).

   The gmxapi protocol to register a gmxapi::MDModule with a gmxapi::IMDRunner
   by passing an gmx::IRestraintPotential to a gmx::MdRunner is described in the
   gmxapi::IMDRunner docs. To exercise it, we need to call gmxapi::IMDRunner::setRestraint(),
   passing a std::shared_ptr<gmxapi::MDModule> argument.

        class NullRestraint : public gmx::IRestraintPotential
        {
            public:
                gmx::PotentialPointData evaluate(gmx::Vector r1,
                                                 gmx::Vector r2,
                                                 double t) override
                {
                    return {};
                }
        };

   Use gmxapi::Module to define an API object class that we can pass around.

        class SimpleApiModule : public gmxapi::MDModule
        {
            public:
                const char *name() override
                {
                    return "NullApiModule";
                }

                // Implement the MDModule protocol.
                std::shared_ptr<gmx::IRestraintPotential> getRestraint() override
                {
                    auto restraint = std::make_shared<NullRestraint>();
                    return restraint;
                }
        };

   C++ client code to run an MD simulation with the custom restraint.

        bool mysim() {
                auto system = gmxapi::fromTprFile(filename);
                std::shared_ptr<gmxapi::Context> context = gmxapi::defaultContext();
                auto runner = system->runner();

                auto session = runner->initialize(context);

                auto module = std::make_shared<SimpleApiModule>();
                session->setRestraint(module);

                gmxapi::Status status;
                status = session->run();
                return status.success();
        };

 */

}      // end namespace gmxapi

#endif // header guard
