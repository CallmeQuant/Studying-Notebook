# Vehicle Routing Problem with Reinforcement Learning
An example to solve (Capacitated) Vehicle Routing Problem. In CVRPT, a number of vehicles with a limited capacity leave depot to serve a set of customers and finally return to the depot. If the load is used up, the vehicle must return back to 
the depot to refill its capacity. 

# Dependencies
You can create an environment with the `requirements.txt` file 

## How to run
To run the default version
```python
python3 main.py
```
To run specific scenario, call the following command
```python
python3 main.py --nodes 10 --test True
```
If you want to use checkpoint, specify a path to the checkpoint folders that contains the networks checkpoints.
For example, in the notebooks, I store it in the folder named "10" in drive.
```python
python3 main.py --nodes 10 --checkpoint 10 --test True
```
To run with different scenarios, change "num_nodes" flag which represents number of cities/customers for example.
```python
python3 main.py --nodes 20 --checkpoint 20 --test True
```

# Notes on VRP (CVRP)
The Vehicle Routing Problem (VRP) involves a scenario where a vehicle or salesman visits various cities/customers, each with a randomly assigned demand ranging from 1 to 9. The vehicle starts with a certain capacity that varies based on the complexity of the problem, such as the maximum capacity.

The VRP uses a specific masking scheme to generate feasible solution space:

  i. If no city has any remaining demand, the tour ends. This implies that the vehicle must return to the depot to complete the tour.
  ii. The vehicle can visit any city, provided it can fully meet the city's demand. This can be modified to allow for partial trips if necessary.
  iii. The vehicle is not allowed to visit the depot more than once consecutively, which helps speed up training.
  iv. The vehicle can only visit the depot twice or more in a row if it has completed its route and is waiting for other vehicles to finish (e.g., training in a minibatch setting).

The dynamic updates in this project are as follows:
i. If a vehicle visits a city, its load changes according to the formula: Load = (Load - Demand_i)+. The demand at the city changes according to: Demand_i = (Demand_i - load)+. This means that the vehicle's load decreases by the city's demand, and the city's demand decreases by the amount of load delivered.

# References
Nazari, Mohammadreza, et al. "Deep Reinforcement Learning for Solving the Vehicle Routing Problem." arXiv preprint arXiv:1802.04240 (2018).
