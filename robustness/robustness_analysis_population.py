import matplotlib.pyplot as plt 
from solver_population import * 
import numpy as np  
import os.path      
import pickle     
import seaborn as sns
import pandas as pd
import multiprocessing
from random import randint

from model_clb_population import model_clb_population, param_references
 


def getParamDistrib(number_points = 0, file_name = ""):    
    rand_samples = []    
    
    region = model_regions[0]   
    if number_points:
        samples = region.points[np.random.choice(region.points.shape[0], number_points, replace=False), :]
    else:
        samples = region.points

    rand_samples.append(samples)

    rand_samples = np.array(rand_samples)

    param_names = [r"$\delta_L$", 
                    r"$\gamma_L$", 
                    r"$n_y$", 
                    r"$\theta_L$", 
                    r"$\eta_x$", 
                    r"$\omega_x$", 
                    r"$m_x$", 
                    r"$\delta_x$",                    
                    r"$\gamma_x$",
                    r"$\theta_x$",
                    r"$\rho_x$",
                    #r"$frac_{toggle}$",
                    #r"$frac_{S}$",
                    #r"$frac_{I}$",
                    #r"$frac_{out}$"]
                    #r"$N_{toggle}$",
                    r"$N_{toggle\ not}$",
                    r"$N_{S}$",
                    r"$N_{I}$",
                    r"$N_{out}$"]
    
    df = pd.DataFrame(rand_samples[0])
    df.columns = param_names

    if file_name:
        df.to_csv(file_name, index=False)
    return df

def plotParamsdf(df=None, number_points = 0):
    if not type(df):
        df = getParamDistrib(number_points)
    
    param_names = [ r"$\gamma_L$", 
                    r"$\eta_x$", 
                    r"$\gamma_x$",
                    r"$\theta_L$", 
                    r"$\omega_x$", 
                    r"$\theta_x$",
                    r"$\delta_L$",  
                    r"$\delta_x$",                                     
                    r"$\rho_x$",
                    r"$n_y$",                   
                    r"$m_x$",
                    #r"$frac_{toggle}$",
                    #r"$frac_{S}$",
                    #r"$frac_{I}$",
                    #r"$frac_{out}$"]
                    #r"$N_{toggle}$",
                    r"$N_{toggle\ not}$",
                    r"$N_{S}$",
                    r"$N_{I}$",
                    r"$N_{out}$"]
    
    #units = [r"$nM/min$", r"$nM/min$", r"$nM/min$", r"$nM^{-1}$", r"$nM^{-1}$", r"$nM^{-1}$", r"$min^{-1}$", r"$min^{-1}$", r"$min^{-1}$", "", "", "", "", "", ""]
    units = [r"$nM/min$", r"$nM/min$", r"$nM/min$", r"$nM^{-1}$", r"$nM^{-1}$", r"$nM^{-1}$", r"$min^{-1}$", r"$min^{-1}$", r"$min^{-1}$", "", "", "", "", "", "", ""]
    

    fig, axes = plt.subplots(4,4)

    for i, (param_name,unit) in enumerate(zip(param_names, units)):
        if param_name:
            ax = axes.flat[i]
            sns.violinplot(data=df[param_name], ax = ax) #,palette="Pastel1")
            ax.set_xticks([])
            #ax.set_xticks([0])
            #ax.set_xticklabels([param_name])    
            if unit:    
                ax.set_ylabel(param_name + " [" + unit + "]")
            else:
                ax.set_ylabel(param_name)

            #ax.set_yscale('log')

    """
    for param_id in range(len(param_names)):
        ax = axes.flat[param_id]

        sns.violinplot(y = param_names[param_id], x="Model id", data=df[[param_names[param_id], "Model id"]], ax = ax) #,palette="Pastel1")
    """ 
    fig=plt.gcf()

    fig.set_size_inches([15,12])
    #plt.savefig(os.path.join('results_robustness', 'params_distrib_sns.pdf'), bbox_inches = 'tight')
    plt.show()


def plot_populations(df):
    df = df[df.columns[-5:]]
    #sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.05})
    sns.pairplot(df.loc[:100], kind='kde')
    plt.show()


def plot_frac(df, box=False):
    df = df[df.columns[-4:]]
    """
    param_names = df.columns
    units = ["", "", "", ""]

    fig, axes = plt.subplots(1, 4, sharey=True)

    for i, (param_name,unit) in enumerate(zip(param_names, units)):
        if param_name:
            ax = axes.flat[i]
            if box:
                sns.boxplot(data=df[param_name], ax = ax)
            else:
                sns.violinplot(data=df[param_name], ax = ax, cut = 0)
            
            #sns.distplot(df[param_name],ax = ax) #,palette="Pastel1")
            #ax.hist(df[param_name], bins=100)
            ax.set_xticks([])
            ax.set_xticks([])
            #ax.set_xticklabels([param_name])    
            if unit:    
                ax.set_ylabel(param_name + " [" + unit + "]")
            else:
                ax.set_ylabel(param_name)
            ax.tick_params(axis='both', which='both', reset=True)
                

    fig.set_size_inches([12,4])
    """

    sns.violinplot(data = df, cut=0, color="#3274a1")
    plt.ylabel('Population size')
    #x1,x2,_,_ = plt.axis()
    #plt.plot([x1, x2], [1,1], 'r--', linewidth=0.5)

    fig = plt.gcf()
    fig.set_size_inches([12,4])

    plt.savefig(os.path.join('results_robustness_population', 'population_distrib_sns.pdf'), bbox_inches = 'tight')
    plt.show()

def test_random_point():
    points = model_regions[0].points
    candidate = tuple(points[randint(0,len(points)-1)])
    model.simulate(candidate, plot_on=True)

if __name__ == "__main__":

    sns.set_style("white")
    #flatui = ['#d9d9d9','#bdbdbd','#969696','#636363']
    #sns.set_palette(flatui)

    #
    # SETTINGS
    #
    read_data = True
    ga_solutions = False
    local_solutions = True

    
    base_paths_opt = [os.path.join("results_optimization_population", "cblb_pop_frac")]


    #
    # END OF SETTINGS
    #

    if read_data:


        #folders = [os.path.join(base_path, "one_bit_model"), os.path.join(base_path, "two_bit_model"), os.path.join(base_path, "three_bit_model")]   
        model = model_clb_population()
        solver = Solver(model)

        model_regions = []

        
        region_files =  []
        for base_path_opt in base_paths_opt:
            if ga_solutions:
                region_files.append(base_path_opt + "ViableSet_IterGA.p")
            if local_solutions:
                for i in range(10):
                    region_files.append(base_path_opt+"_Region0ViableSet_Iter" + str(i+1) + ".p")

        
        viablePoints = []   
        for region_file in region_files: 
            try:
                viablePointsRegion = pickle.load(open(region_file, "rb"))   
                print(len(viablePointsRegion))   
                viablePoints.extend(viablePointsRegion)
            except:
                print("Load of " + region_file + " failed!")
        print("Number of points: ",len(viablePoints))
        region = Region(viablePoints, model, "region")              
        model_regions.append(region)                                                                        
        
    


 
    df = getParamDistrib(file_name=os.path.join("results_robustness_population", "params_population.csv"))
    #df = pd.read_csv(os.path.join("results_robustness_population", "params_population.csv"))

    #plotParamsdf(df)
    #plot_populations(df)
    plot_frac(df)

    print(np.round(df.mean(),2))
    print(np.round(df.std(),2))

    #test_random_point()
    