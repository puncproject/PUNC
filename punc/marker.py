from __future__ import print_function
import dolfin as df
import numpy as np


def mark_exterior_boundaries(facet_f, n_components, domain_info):
    """
    Marks the exteror boundaries of the simulation domain

    Args:
        facet_f:  Facet function.

    returns:
        facet_f: marked facets of the exterior boundaries of the domain
    """
    d = int(len(domain_info)/2)
    for i in range(2*d):
        boundary = 'near((x[i]-l), 0, tol)'
        boundary = df.CompiledSubDomain(boundary,
                                        i = i%d,
                                        l = domain_info[i],
                                        tol = 1E-8)
        boundary.mark(facet_f, (n_components+i))
    return facet_f
