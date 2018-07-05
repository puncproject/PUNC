// Copyright (C) 2017, Sigvald Marholm, Diako Darian and Mikael Mortensen
//
// This file is part of PUNC.
//
// PUNC is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC. If not, see <http://www.gnu.org/licenses/>.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/fem/DirichletBC.h>
#include <algorithm>

namespace dolfin
{
  struct myclass {
    bool operator() (int i,int j) { return (i<j);}
  } myobject;

  void apply(GenericMatrix& A, const std::vector<dolfin::la_index> &ind0)
  {
    std::vector<std::size_t> neighbors;
    std::vector<double> values;
    std::vector<std::size_t> surface_neighbors;
    std::vector<dolfin::la_index> zero_row;
    std::size_t self_index;
    std::vector<std::size_t> ind;
    std::vector<std::vector<std::size_t> > allneighbors;
    int m;

    for (std::size_t i = 0; i < ind0.size(); i++)
        ind.push_back(ind0[i]);
    
    std::sort (ind.begin(), ind.end(), myobject );

    for (std::size_t i = 0; i < ind.size(); i++)
    { 
      if (ind[i] == ind0[0])
        continue;
      std::size_t row = ind[i];
      A.getrow(row, neighbors, values);
      allneighbors.push_back(neighbors);
    }
    A.zero(ind0.size()-1, ind0.data()+1);

    std::size_t count = 0;
    for (std::size_t i = 0; i < ind.size(); i++)
    {       
      if (ind[i] == ind0[0])
        continue;
      
      std::size_t row = ind[i];
      surface_neighbors.clear();
      values.clear();
      for (std::size_t j = 0; j < allneighbors[count].size(); j++)
      {
         std::size_t n = allneighbors[count][j]; 
         if (std::binary_search(ind.begin(), ind.end(), n))
         {
           surface_neighbors.push_back(n);
           values.push_back(-1.0);
         }
      }
      for (std::size_t j = 0; j < surface_neighbors.size(); j++)
      {
        if (surface_neighbors[j] == row)
        {
          self_index = j;
          break;
        }
      }
      std::size_t num_of_neighbors = surface_neighbors.size()-1;
      values[self_index] = num_of_neighbors;
      A.setrow(row, surface_neighbors, values);
      count++;
    }
    A.apply("insert");  
  }

  PYBIND11_MODULE(SIGNATURE, m)
  {
    m.def("apply", &apply);
  }
}        

