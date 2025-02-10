from evotorch import Problem
from evotorch.algorithms import MAPElites
import torch
from evotorch.operators import GaussianMutation
from typing import List
import subprocess
import logging

NUM_EXPERIMENTS = 5

class AggregationProblem(Problem):
    def __init__(self, ):
        super().__init__(objective_sense="max", objective_func=run_simulation, solution_length=8, dtype=torch.float32, 
                         device="cpu", eval_data_length=2, seed=42, num_actors=1, bounds=[0., 1.])

def run_simulation(weights: List[float]):
    fitness = 0
    msd = 0
    density = 0

    weights = weights.clone()
    weights[1:4] *= 0.01
    weights[5:8] *= 0.01

    for i in range(NUM_EXPERIMENTS):
        output = subprocess.check_output(['./main', str(weights[0].item()), str(weights[1].item()),
                                           str(weights[2].item()), str(weights[3].item()),  str(weights[4].item()), str(weights[5].item()),
                                            str(weights[6].item()), str(weights[7].item()), "0"], text=True).split()
        fitness += int(output[0])
        msd += float(output[1])
        density += float(output[2])
        
    fitness /= NUM_EXPERIMENTS
    msd /= NUM_EXPERIMENTS
    density /= NUM_EXPERIMENTS

    logging.info(f"     Solution = {weights}, fitness = {fitness}, msd = {msd}, density = {density}")
    return torch.tensor([fitness, msd, density])


if __name__ == "__main__":

    problem = AggregationProblem()

    feature_grid = MAPElites.make_feature_grid(
        lower_bounds=torch.tensor([0., 0]),
        upper_bounds=torch.tensor([2000., 20]),
        num_bins=10
    )

    mutation = GaussianMutation(problem=problem, stdev=0.1)
    operators = [mutation,]
    searcher = MAPElites(problem, operators=operators, feature_grid=feature_grid, re_evaluate=False)

    num_generations = 20

    data = torch.zeros((num_generations, len(searcher.population), 3))
    solutions = torch.zeros((num_generations, len(searcher.population), 8))

    logging.basicConfig(
        filename="results/res.log",
        encoding="utf-8",
        filemode="w",
        level=logging.INFO
    )
    logging.info('STARTING OPTIMIZATION')

    for generation in range(50):
        searcher.step()

        # save solutions and data to files
        fitnesses = []
        for i, solution in enumerate(searcher.population):
            if searcher.filled[i]:
                cluster = float(solution.evals[0])
                msd = float(solution.evals[1])
                density = float(solution.evals[2])
                data[generation, i, 0] = cluster
                data[generation, i, 1] = msd
                data[generation, i, 2] = density
                solutions[generation, i] = solution.values
                fitnesses.append(cluster)

        logging.info(f"GENERATION: {generation + 1}: best fitness = {max(fitnesses)}, mean fitness = {sum(fitnesses)/len(fitnesses)}")

        torch.save(data, f='results/data.pt')
        torch.save(solutions, f="results/solutions.pt")