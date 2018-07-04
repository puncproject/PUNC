#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/ArrayView.h>

namespace dolfin
{

void addrow(GenericMatrix& B, GenericMatrix& Bc,
            const pybind11::array_t<std::size_t> &pycols,
            const pybind11::array_t<double> &pyvals,
            size_t replace_row, const FunctionSpace& V)
{
  Timer timer("Add row with new sparsity to matrix");

  std::vector<std::size_t> cols = pycols.cast<std::vector<std::size_t>>();
  std::vector<double> vals = pyvals.cast<std::vector<double>>();

  std::shared_ptr<TensorLayout> layout;
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps.push_back(V.dofmap().get());
  
  const Mesh& mesh = *(V.mesh());  
  layout = Bc.factory().create_layout(mesh.mpi_comm(), 2);
  dolfin_assert(layout);

  std::vector<std::shared_ptr<const IndexMap>> index_maps;
  for (std::size_t i = 0; i < 2; i++)
  {
    dolfin_assert(dofmaps[i]);
    index_maps.push_back(dofmaps[i]->index_map());
  }
  layout->init(index_maps, TensorLayout::Ghosts::UNGHOSTED);

  SparsityPattern& new_sparsity_pattern = *layout->sparsity_pattern();
  new_sparsity_pattern.init(index_maps);

  // With the row-by-row algorithm used here there is no need for
  // inserting non_local rows
  const std::size_t primary_dim = new_sparsity_pattern.primary_dim();
  const std::size_t primary_codim = primary_dim == 0 ? 1 : 0;
  const std::pair<std::size_t, std::size_t> primary_range
    = index_maps[primary_dim]->local_range();
  const std::size_t secondary_range
    = index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
  const std::size_t diagonal_range
    = std::min(primary_range.second, secondary_range);
  const std::size_t m = diagonal_range - primary_range.first;

  // Declare some variables used to extract matrix information
  std::vector<size_t> columns;
  /* pybind11::array_t<size_t> columns; */
  std::vector<double> values;

  // Hold all values of local matrix
  std::vector<double> allvalues;

  // Hold column id for all values of local matrix
  std::vector<dolfin::la_index> allcolumns;

  // Hold accumulated number of cols on local matrix
  std::vector<dolfin::la_index> offset(m + 1);

  offset[0] = 0;
  std::vector<ArrayView<const dolfin::la_index>> dofs(2);
  std::vector<std::vector<dolfin::la_index>> global_dofs(2);

  global_dofs[0].push_back(0);
  // Iterate over rows
  for (std::size_t i = 0; i < (diagonal_range - primary_range.first); i++)
  {
    // Get row and locate nonzeros. Store non-zero values and columns
    // for later
    const std::size_t global_row = i + primary_range.first;
    std::size_t count = 0;
    global_dofs[1].clear();
    columns.clear();
    values.clear();
    if (global_row == replace_row)
    { 
      if (MPI::rank(mesh.mpi_comm()) == 0)
      {
        columns = cols;
        values = vals;
      }
    }
    else
    {
      B.getrow(global_row, columns, values);
    }
    for (std::size_t j = 0; j < columns.size(); j++)
    {
      // Store if non-zero or diagonal entry.
      if (std::abs(values[j]) > DOLFIN_EPS || columns[j] == global_row)
      {
        global_dofs[1].push_back(columns[j]);
        allvalues.push_back(values[j]);
        allcolumns.push_back(columns[j]);
        count++;
      }
    }
    global_dofs[0][0] = global_row;
    offset[i + 1] = offset[i] + count;
    dofs[0].set(global_dofs[0]);
    dofs[1].set(global_dofs[1]);
    new_sparsity_pattern.insert_global(dofs);
  }

  // Finalize sparsity pattern
  new_sparsity_pattern.apply();

  // Create matrix with the new layout
  Bc.init(*layout);

  // Put the values back into new matrix
  for (std::size_t i = 0; i < m; i++)
  {
    const dolfin::la_index global_row = i + primary_range.first;
    Bc.set(&allvalues[offset[i]], 1, &global_row,
           offset[i+1] - offset[i], &allcolumns[offset[i]]);
  }
  Bc.apply("insert");
}
PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("addrow", &addrow);
}
}
