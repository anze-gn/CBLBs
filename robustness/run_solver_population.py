from solver_population import Solver
from model_clb_population import model_clb_population, param_references
import os
import numpy as np

if __name__ == "__main__":

    filename = os.path.join("results_optimization_population", "cblb_pop_frac")
    
    #model = BioProc(np.array(["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation", "protein_degradation", "Kd","hill", "protein_production", "protein_degradation", "Kd", "hill"]), model_mode=three_bit_processor_ext, parameter_values=param_values, avg_dev=30)                                         
    model = model_clb_population()
    solver = Solver(model, populationSize=1000)                         
    solver.run(filename, maxDepth=1) #do not cluster