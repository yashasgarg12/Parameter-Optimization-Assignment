from pretty_html_table import build_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_results(results, filename):
    table = build_table(results, 'blue_light')
    with open(filename, 'w') as f:
        f.write(table)
        print(f"Results saved to {filename}")

def convergence_graph(results, filename):
    #plot iteration vs fitness graph
    df = pd.DataFrame(results)
    df['Iteration'] = df.index
    df['Fitness'] = df['Fitness'].astype(float)
    df.plot(x='Iteration', y='Fitness', kind='line')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Convergence Graph')
    plt.savefig(filename)
    print(f"Convergence graph saved to {filename}")