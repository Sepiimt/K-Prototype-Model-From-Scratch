import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import datetime as dt
import os

class K_Prototype:
    def __init__(self):
        #> For Plot Function
        self.best_cluster_points = None
        self.K_history = []
        self.cost_per_k_history=[]
        #> For Prototype Initilizer & Prototype Checker Functions
        self.rng = np.random.default_rng()
        #> For Wide Range of Uses
        self.numerical_prototypes = None
        self.categorical_prototypes = None
        #> For Timer Function
        self.start_time_stamp = None
        self.finish_time_stamp = None
        
    def random_prototypes_initializer(self, numerical_array, categorical_array, K):
        random_rowss = self.rng.choice(len(numerical_array), size=K, replace=False)
        numerical_prototypes = numerical_array[random_rowss]
        categorical_prototypes = categorical_array[random_rowss]
        return numerical_prototypes, categorical_prototypes

    def numerical_distance(self, numerical_array, numerical_prototypes):
        diff = numerical_array - numerical_prototypes[:, np.newaxis]
        return np.sum(diff**2, axis=-1)

    def categorical_mismatch(self, categorical_array, categorical_prototypes):
        mismatches = categorical_array != categorical_prototypes[:, np.newaxis]
        return np.sum(mismatches, axis=-1)

    def total_cost_computer(self, gamma, distance_results_array, mismatch_results_array):
        # --- Computing Distance to Each Prototype --- 
        total_distance_array = distance_results_array + gamma * mismatch_results_array
        return total_distance_array

    def dedicated_cost_computer(self, total_distance_array):
        # --- Computing Cost Of All Prototypes --- 
        cost_value = (np.min(total_distance_array, axis=0)).sum()
        return cost_value

    def data_clusterer(self, total_distance_array):
        #> Format: Returning a 1D array (example: [0, 1, 0, 2, 1])- 
        #> -where each value represents the assigned cluster for the corresponding data point.
        distinguisher_labels_array = np.argmin(total_distance_array, axis=0)
        return distinguisher_labels_array
    
    def sub_prototype_computer_for_numerical(self,weight_matrix, numerical_array):
        # --- Calculating Cluster Counts and Numerical Prototypes ---
        cluster_counts = np.sum(weight_matrix, axis=1, keepdims=True)
        numerical_prototypes = np.dot(weight_matrix, numerical_array) / np.maximum(cluster_counts, 1)
        return cluster_counts, numerical_prototypes

    def sub_prototype_computer_for_categorical(self, weight_matrix, categorical_array):
        categorical_prototypes = []
        # --- Looping Through Each Cluster ---
        for mask in weight_matrix:
            # --- Seperating Mask Per Cluster ---
            cluster_data = categorical_array[mask.astype(bool)]
            # --- Handling Empty Clusters ---
            if cluster_data.size == 0:
                categorical_prototypes.append(np.full(categorical_array.shape[1], fill_value=None))
                continue
            # --- Finding Mode for All Columns Simultaneously ---
            cluster_modes = stats.mode(cluster_data, axis=0, keepdims=False).mode
            categorical_prototypes.append(cluster_modes)
        # --- Return ---
        return np.array(categorical_prototypes)

    def prototypes_computer(self, numerical_array, categorical_array,
                            total_distance_array, K, update_values=False, 
                            return_values=False):
        distinguisher_labels_array = self.data_clusterer(total_distance_array)
        weight_matrix = (distinguisher_labels_array == np.arange(K)[:, np.newaxis])
        #--- Numerical Prototypes ---
        cluster_counts, numerical_prototypes = self.sub_prototype_computer_for_numerical(weight_matrix, numerical_array)
        # --- Categorical Prototypes ---
        categorical_prototypes = self.sub_prototype_computer_for_categorical(weight_matrix, categorical_array)
        # --- Empty Cluster Check ---
        numerical_prototypes, categorical_prototypes = self.empty_prototype_checker(cluster_counts, numerical_prototypes, 
                                                                                    categorical_prototypes, numerical_array, 
                                                                                    categorical_array)
        # --- Attribute Update ---
        if update_values:
            self.prototype_information_updater(numerical_prototypes, categorical_prototypes)
        # --- Returning Values ---
        if return_values:
            return numerical_prototypes, categorical_prototypes

    def empty_prototype_checker(self, cluster_counts, numerical_prototypes, 
                                categorical_prototypes, numerical_array, categorical_array):
        #> If There Are Empty Prototypes, We Re-Initialize Them Randomly
        # --- Finding Empty Prototypes ---
        empty_prototype_indices = np.where(cluster_counts.flatten() == 0)[0]
        # --- Stop Guard ---
        if len(empty_prototype_indices) == 0:
            return numerical_prototypes, categorical_prototypes
        # --- Re-Initializing Empty Prototypes ---
        random_rows = self.rng.choice(len(numerical_array), size=len(empty_prototype_indices), replace=False)
        numerical_prototypes[empty_prototype_indices] = numerical_array[random_rows]
        categorical_prototypes[empty_prototype_indices] = categorical_array[random_rows]
        # --- Return ---
        return numerical_prototypes, categorical_prototypes

    def prototype_information_updater(self, numerical_prototypes, categorical_prototypes):
        self.numerical_prototypes = numerical_prototypes
        self.categorical_prototypes = categorical_prototypes

    def best_init_prototype_points(self, numerical_array, categorical_array, K, gamma, n_attempts):
        # --- If Multiple Attempts Are Specified ---
        if n_attempts is not None and n_attempts > 0:
            best_round_cost=np.inf
            best_numerical_prototypes, best_categorical_prototypes = None, None
            for _ in range(n_attempts):
                    numerical_prototypes, categorical_prototypes = self.random_prototypes_initializer(numerical_array, categorical_array, K)
                    # --- Getting Distance & Mismatches Arrays ---
                    distance_results_array = self.numerical_distance(numerical_array, numerical_prototypes)
                    mismatch_results_array = self.categorical_mismatch(categorical_array, categorical_prototypes)
                    # --- Calculating Cost Value ---
                    total_distance_array = self.total_cost_computer(gamma, distance_results_array, mismatch_results_array)
                    cost_value = self.dedicated_cost_computer(total_distance_array)
                    # --- Updating On Criteria Of Better Results ---
                    if best_round_cost > cost_value:
                        best_round_cost = cost_value
                        best_numerical_prototypes = numerical_prototypes.copy()
                        best_categorical_prototypes = categorical_prototypes.copy()
            # --- Return ---
            if best_numerical_prototypes is not None and best_categorical_prototypes is not None:
                return best_numerical_prototypes, best_categorical_prototypes
            else:
                raise ValueError("Best Prototypes Could Not Be Determined!")
        # --- If No Multiple Attempts Are Specified ---
        else:
            numerical_prototypes, categorical_prototypes = self.random_prototypes_initializer(numerical_array, categorical_array, K)
            # --- Return ---
            return numerical_prototypes, categorical_prototypes

    def prototype_converger(self, numerical_array, categorical_array, 
                            numerical_prototypes, categorical_prototypes, 
                            K, gamma, max_iterations=1000, return_prototypes=False, return_cost=False):
        # --- Setting Default Values ---
        n_round, convergance = 0, False
        # --- The Loop ---
        while convergance==False and n_round < max_iterations:
            last_numerical_prototypes, last_categorical_prototypes = numerical_prototypes.copy(), categorical_prototypes.copy()
            # --- Getting Distance & Mismatches Arrays ---
            distance_results_array = self.numerical_distance(numerical_array, numerical_prototypes)
            mismatch_results_array = self.categorical_mismatch(categorical_array, categorical_prototypes)
            # --- Calculating Cost Value ---
            total_distance_array = self.total_cost_computer(gamma, distance_results_array, mismatch_results_array)
            # --- Updating Prototypes ---
            numerical_prototypes, categorical_prototypes = self.prototypes_computer(numerical_array, categorical_array, 
                                                                                    total_distance_array, K, update_values=False, 
                                                                                    return_values=True)
            # --- Counting Iteration ---
            n_round += 1
            # --- Checking Convergance ---
            convergance_criteria = (np.array_equal(last_numerical_prototypes, numerical_prototypes) & 
                                    np.array_equal(last_categorical_prototypes, categorical_prototypes))
            if convergance_criteria:
                cost_value = self.dedicated_cost_computer(total_distance_array)
                convergance = True
        # --- Return ---
        if return_prototypes and return_cost:
            return numerical_prototypes, categorical_prototypes, cost_value
        if return_prototypes:
            return numerical_prototypes, categorical_prototypes
        if return_cost:
            return cost_value

    def elbow_plotting_information_updater(self, K, cost_value):
        self.K_history.append(K)
        self.cost_per_k_history.append(cost_value)

    def elbow_plotter_visual(self, K_history, cost_per_k_history):
        plt.figure()
        plt.plot(K_history, cost_per_k_history, label="Elbow Plot", color="lime")
        plt.xlabel("K")
        plt.ylabel("Cost")
        plt.title("Cost Change Per K")
        plt.legend()
        plt.grid(True)
        # --- Ensuring Save Directory Exists ---
        save_dir = r"..\results\figures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # --- Saving The Plot ---
        plt.savefig(os.path.join(save_dir, "cost_per_k.png"), dpi=300)
        plt.show()
        plt.close()

    def elbow_plotter(self, numerical_array, 
                      categorical_array, gamma = None, 
                      max_k = 10, n_attempts_for_best_init_prototypes = None, 
                      max_convergance_iterations=1000, timer = True):
        # --- Setting Gamma Value ---
        if gamma is None:
            gamma = self.set_gamma(numerical_array)
        # --- Resetting Values ---
        self.K_history,  self.cost_per_k_history = [], []
        # --- Timer ---
        if timer:
            self.fit_timer(status=1)
        # --- Outer Loop ---
        for K in range(2, max_k+1):
            # --- Initializing Prototypes ---
            numerical_prototypes, categorical_prototypes = self.best_init_prototype_points(numerical_array,
                                                                                           categorical_array,
                                                                                           K, gamma, 
                                                                                           n_attempts_for_best_init_prototypes)
            # --- Updating Prototypes Until Convergence ---
            cost_value = self.prototype_converger(numerical_array, categorical_array, 
                                                  numerical_prototypes, categorical_prototypes, 
                                                  K, gamma, max_convergance_iterations,
                                                  return_prototypes=False, return_cost=True)
            # --- Updating Elbow Plotting Information ---
            self.elbow_plotting_information_updater(K, cost_value)
        # --- Timer ---
        if timer:
            self.fit_timer(status=0)
        # --- Visualizing the Plot ---
        self.elbow_plotter_visual(self.K_history, self.cost_per_k_history)

    def set_gamma(self, numerical_array):
        self.gamma = 0.5 * numerical_array.std()
        return self.gamma
    
    def fit_k_false_value_check(self, K, n_data_points):
        if K==1 or K==0 or K<0 or not isinstance(K, int) or K > n_data_points:
            raise ValueError(f"'K' Value Cannot Be {K}!")

    def fit_timer(self, status):
        if status == 1 :
            self.start_time_stamp = dt.datetime.now()
        elif status == 0 :
            self.finish_time_stamp = dt.datetime.now()
            print(f"Time Taken: {self.finish_time_stamp - self.start_time_stamp}")
        
    def fit(self, numerical_array, categorical_array, 
            K, gamma = None, n_attempts_for_best_init_prototypes = None, 
            max_convergance_iterations=1000, timer = True):
        # --- Stop Guard ---
        self.fit_k_false_value_check(K, len(numerical_array))
        # --- Setting Gamma Value ---
        if gamma is None:
            gamma = self.set_gamma(numerical_array)
        # --- Resetting Values ---
        self.prototype_information_updater(numerical_prototypes = None, categorical_prototypes = None)
        # --- Timer ---
        if timer:
            self.fit_timer(status=1)
        # --- Initializing Prototypes ---
        numerical_prototypes, categorical_prototypes = self.best_init_prototype_points(numerical_array,
                                                                                           categorical_array,
                                                                                           K, gamma, n_attempts_for_best_init_prototypes)
        # --- Updating Prototypes Until Convergence ---
        numerical_prototypes, categorical_prototypes = self.prototype_converger(numerical_array, categorical_array, 
                                                numerical_prototypes, categorical_prototypes, 
                                                K, gamma, max_convergance_iterations,
                                                return_prototypes=True, return_cost=False)
        # --- Updating Fit Information ---
        self.prototype_information_updater(numerical_prototypes, categorical_prototypes)
        if timer:
            self.fit_timer(status=0)
        return numerical_prototypes, categorical_prototypes