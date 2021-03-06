/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2018,2019, by the GROMACS development team, led by
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

#ifndef GMX_GMXPREPROCESS_VSITE_PARM_H
#define GMX_GMXPREPROCESS_VSITE_PARM_H

#include "gromacs/utility/arrayref.h"

class PreprocessingAtomTypes;
struct gmx_moltype_t;
struct t_atoms;
struct InteractionsOfType;

int set_vsites(bool bVerbose, t_atoms *atoms,  PreprocessingAtomTypes *atype,
               gmx::ArrayRef<InteractionsOfType> plist);
/* set parameters for virtual sites, return number of virtual sites */

void set_vsites_ptype(bool bVerbose,  gmx_moltype_t *molt);
/* set ptype to VSite for virtual sites */

/*! \brief Clean up the bonded interactions
 *
 * Throw away all obsolete bonds, angles and dihedrals.
 * Throw away all constraints. */
void clean_vsite_bondeds(gmx::ArrayRef<InteractionsOfType> ps, int natoms, bool bRmVSiteBds);

#endif
