def vrp_solver(vehicles, capacity, items = None, cost_matrix = None, dir = None):
  # check if given cost matrx
  if cost_matrix is None:
    if dir is None:
      raise ValueError('If cost matrix has not given, you must pass in cost data directory')
    else:
      cost_matrix = pd.read_csv(dir)

  assert cost_matrix.shape[0] == cost_matrix.shape[0], 'Cost matrix should be a square matrix'
  if items is None:
    print('Items is set to None. Set the name of items based on items length')
    items = [i for i in range(cost_matrix.shape[0])]


  row, col = cost_matrix.shape
  res = []
  res_name = []
  M = 10000
  res_df = pd.DataFrame()
  vehicles = vehicles

  # Define problem
  problem = LpProblem('TravellingSalesmanProblem', LpMinimize)

  # Define variables
  # Decision variable X & Y for truck route
  X = LpVariable.dicts('X', ((i, j, k) for i in items for j in items for k in range(vehicles)), lowBound=0, upBound=1, cat='Integer')
  Y = LpVariable.dicts('y', ((i, k) for i in items for k in range(vehicles)), lowBound=0, upBound=1, cat='Integer')

  # subtours elimination - Order of each node
  U = LpVariable.dicts('U', ((i, k) for i in items for k in range(vehicles)), lowBound=0, cat='Integer')

  # Decision variable T for truck arrival time
  T = LpVariable.dicts('T', ((i,k) for i in items for k in range(vehicles)), lowBound=0, cat='Float')

  # Objective function
  problem += lpSum(T[i, k] for i in items for k in range(vehicles))

  # Constraint with vehicles
  for k in range(vehicles):
    problem += lpSum(Y[i, k] for i in items) <= capacity
    for i in items:
      problem += lpSum(X[i, i, k] == 0) # Prevent staying at one node
      if i == 0:
        problem += lpSum(T[i, k] == 0)

  for i in items:
    problem += lpSum(Y[i, k] for k in range(vehicles)) == 1
    for k in range(vehicles):
      problem += lpSum(X[i, j, k] for j in items) == Y[i, k]
      problem += lpSum(X[j, i, k] for j in items) == Y[i, k]
      if (i == 0):
        # At start node, vehicle k can travel from i to j or not
        for k in range(vehicles):
          problem += lpSum(X[i, j, k] for j in items) <= 1
          problem += lpSum(X[j, i, k] for j in items) <= 1

  for i in items:
    for j in items:
      for k in range(vehicles):
        if i != j and (j != 0):
          problem += T[j, k] >= T[i, k] + cost_matrix[i][j] - M*(1-X[i, j, k]) # Calculating time of arrival at each node
        if i != j and (i != 0) and (j != 0):
          problem += U[i, k]  <=  U[j, k] + M * (1 - X[i, j, k]) - 1 # sub-tour elimination for truck



  status = problem.solve(PULP_CBC_CMD())
  flag = 0
  for var in problem.variables():
    if (problem.status == 1):
      flag += 1
      if (var.value() != 0):
        res.append(var.value())
        res_name.append(var.name)

  if flag == len(problem.variables()):
    print('Optimal solution found for all variables')
  res_df['Variable Name'] = res_name
  res_df['Variable Value'] = res
  return (problem.status, problem.objective.value(), res_df)
