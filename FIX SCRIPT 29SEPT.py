# %%
# Import Library
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from pymoo.indicators.hv import HV

# %%
# Load datasets (example datasets, replace with your actual data loading)
processing_times = pd.read_csv('processing_times.csv', index_col=0)
due_dates = pd.read_csv('due_dates.csv', index_col=0)
lateness_costs = pd.read_csv('tardiness_costs.csv', index_col=0)

# Number of jobs and machines
num_jobs = processing_times.shape[0]
num_machines = processing_times.shape[1]

# %%
# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Two objectives: makespan and total lateness cost
creator.create("Individual", list, fitness=creator.FitnessMin)

# %%
# Initialize toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_job", np.random.permutation, num_jobs)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_job, num_machines)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %%
# Evaluation function
def evaluate(individual):
    schedule = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))

    for machine_idx in range(num_machines):
        for job_idx in range(num_jobs):
            job = individual[machine_idx][job_idx]
            if job_idx == 0:
                start_time = 0
            else:
                previous_job = individual[machine_idx][job_idx - 1]
                start_time = completion_times[previous_job, machine_idx]

            if machine_idx > 0:
                prev_machine_end_time = completion_times[job, machine_idx - 1]
                start_time = max(start_time, prev_machine_end_time)

            end_time = start_time + processing_times.iloc[job, machine_idx]
            schedule[job, machine_idx] = start_time
            completion_times[job, machine_idx] = end_time

    # Total lateness cost
    total_lateness_cost = 0
    for job in range(num_jobs):
        for machine_idx in range(num_machines):
            if job_idx == 0:  # First job on any machine
                planned_start_time = 0
            elif machine_idx == 0:
                planned_start_time = schedule[job, machine_idx] 
            else:
                planned_start_time = completion_times[job, machine_idx - 1] 
            planned_completion_time = planned_start_time + due_dates.iloc[job, machine_idx]
            lateness = max(0, completion_times[job, machine_idx] - planned_completion_time)
            total_lateness_cost += lateness* lateness_costs.iloc[job, machine_idx]

    # Makespan is the maximum completion time
    makespan = np.max(completion_times)

    return makespan, total_lateness_cost 

# %%
# Genetic operators

toolbox.register("mate", tools.cxUniform, indpb=0.7) 
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3) 
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# %%
def calculate_hypervolume(pop, ref_point):
    fitnesses = np.array([ind.fitness.values for ind in pop])
    hv = HV(ref_point=ref_point)
    return hv(fitnesses)

# %%
def generate_timeline(individual):
    schedule = []
    completion_times = np.zeros((num_jobs, num_machines))

    for machine_idx in range(num_machines):
        machine_schedule = []
        start_time = 0
        for job_idx in range(num_jobs):
            job = individual[machine_idx][job_idx]
            processing_time = processing_times.iloc[job, machine_idx]

            if job_idx == 0:
                start_time = 0
            else:
                previous_job = individual[machine_idx][job_idx - 1]
                start_time = completion_times[previous_job, machine_idx]

            if machine_idx > 0:
                prev_machine_end_time = completion_times[job, machine_idx - 1]
                start_time = max(start_time, prev_machine_end_time)

            end_time = start_time + processing_time
            completion_times[job, machine_idx] = end_time

            # Calculate planned start time, planned completion time and lateness cost
            if machine_idx == 0:
                planned_start_time = start_time 
            else:
                planned_start_time = completion_times[job, machine_idx - 1] 

            planned_completion_time = planned_start_time + due_dates.iloc[job, machine_idx]
            lateness = max(0, end_time - planned_completion_time)
            lateness_cost = lateness * lateness_costs.iloc[job, machine_idx]

            machine_schedule.append({
                'Job': job,
                'Machine': machine_idx,
                'Start': start_time,
                'End': end_time,
                'Planned Start': planned_start_time,
                'Planned Completion': planned_completion_time,
                'Lateness Cost': lateness_cost
            })
            start_time = end_time
        schedule.append(machine_schedule)

    # Generate timeline
    timeline = ""
    for machine_idx in range(num_machines):
        timeline += f"Machine {machine_idx}:\n"
        for job in schedule[machine_idx]:
            timeline += f"  Job {job['Job']}: Start={job['Start']} End={job['End']} "
            timeline += f"Planned Start={job['Planned Start']} "
            timeline += f"Planned Completion={job['Planned Completion']} "
            timeline += f"Lateness Cost={job['Lateness Cost']}\n"
        timeline += "\n"


    return timeline

# %%
def main():
    # Create initial population
    pop = toolbox.population(n=300)  # Increase population size for better exploration
    
    # Define the hall of fame to store the best individuals
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    # Define the statistics to be gathered
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Create a list to store all generations' data
    all_generations_data = []
    
    # Reference point for hypervolume calculation
    ref_point = [1e6, 1e6]  # Example reference point, adjust as needed
    
    # List to store hypervolume values
    hypervolumes = []
    
    # Run the algorithm with NSGA-II
    for gen in range(150):  # Increase the number of generations
        
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=291, lambda_=300, cxpb=0.7, mutpb=0.3, 
                                                 ngen=1, stats=stats, halloffame=hof, verbose=True)
        
        # Calculate hypervolume for the current population
        hypervol = calculate_hypervolume(pop, ref_point)
        hypervolumes.append({"Generation": gen, "Hypervolume": hypervol}) 
        
        # Sort the population into Pareto fronts
        pareto_fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
        
        # Collect all individuals and their fitness values for this generation
        generation_data = []
        for front_idx, front in enumerate(pareto_fronts):
            for ind_idx, ind in enumerate(front):
                generation_data.append({
                    "Generation": gen,
                    "Individual": ind_idx,
                    "Makespan": ind.fitness.values[0],
                    "Total Tardiness Cost": ind.fitness.values[1],
                    "Pareto Front": front_idx
                })
        
        # Append the generation data to the list
        all_generations_data.extend(generation_data)

        # Generate timeline for the best individual
        best_individual = hof[0]
        timeline = generate_timeline(best_individual)
        print(timeline)
    
    # Convert the list to a DataFrame
    all_generations = pd.DataFrame(all_generations_data)
    
    # Save all generations to CSV
    all_generations.to_csv('all_generations_fitness.csv', index=False)
    
    # Convert hypervolume list to a DataFrame
    hypervol_df = pd.DataFrame(hypervolumes)
    
    # Save hypervolume data to CSV
    hypervol_df.to_csv('hypervolumes.csv', index=False)

    return pop, stats, hof, all_generations, hypervol_df

# %%
if __name__ == "__main__":
    pop, stats, hof, all_generations, hypervolumes = main()
    print("All generations saved to 'all_generations_fitness.csv'")
    print("Hypervolumes saved to 'hypervolumes.csv'")
    best_individual = hof[0]
    print(f'Best individual: {best_individual}')
    print(f'Best fitness: {best_individual.fitness.values}')


