import pandas as pd
from gen_sample import gen_random_params
from gen_table import save_results, convergence_graph
from fitness_function import fitness
from sklearn.model_selection import train_test_split

def main():
    #take input from user for iterations 
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


    iterations = int(input("Enter the number of iterations: "))
    results = {'iteration':[], 'C':[], 'kernel':[], 'gamma':[], 'fitness':[]}
    best_accuracy = -10
    best_c = 0
    best_kernel = ''
    best_gamma = 0
    for i in range(iterations):
        C, kernel, gamma = gen_random_params()
        #call fitness function
        fitness_value = fitness(X_train, y_train, X_test, y_test, C = C, kernel = kernel, gamma = gamma)
        if fitness_value > best_accuracy:
            best_accuracy = fitness_value
            best_c = C
            best_kernel = kernel
            best_gamma = gamma
            if i % 10 == 0 or i == iterations - 1:
                print(f"Iteration: {i}, Fitness: {best_accuracy}, c: {best_c}, kernel: {best_kernel}, gamma: {best_gamma}")
                results['iteration'].append(i)
                results['C'].append(best_c)
                results['kernel'].append(best_kernel)
                results['gamma'].append(best_gamma)
                results['fitness'].append(best_accuracy)


    #save results to a file
    save_results(results, 'results.html')
    #plot convergence graph
    convergence_graph(results, 'convergence.png')


if __name__ == '__main__':
    main()
        
