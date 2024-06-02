# Script to analyze onnx model memory profiler summary

import os
import sys

import numpy
import pandas

from matplotlib import pyplot as plt

from pathlib import Path

root = Path(__file__).parents[2].resolve()
workspace = Path(__file__).parent.resolve()

filepath = os.path.join(root, 'results', 'onnxProfile', 'logs', 'sdxlt_unet_summary_analysis.csv')
save_directory = os.path.join(root, 'results', 'onnxProfile', 'plots')

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

table = pandas.read_csv(filepath)

output_memory = table['Output Memory (in MB)'][:-1].to_numpy(dtype=numpy.int64)
total_memory = table['Memory (in MB)'][:-1].to_numpy(dtype=numpy.int64)

main_memory_total_memory_accesses = 0
memory_usage = {}
consecutive_output_memory_usage = {}
optimized_memory_usage = {}
histogram_dict = {}
ordered_histogram_dict = {}

try:
    model_name = sys.argv[1]
    threshold = int(sys.argv[2])

except IndexError:
    model_name = 'Test'
    threshold = 0

# optimize operators having consecutive same output memory size
for i, element in enumerate(total_memory):
    if memory_usage.get(str(element), None) is None:
        memory_usage[str(element)] = 1
        consecutive_output_memory_usage[str(element)] = 0
    
    else:
        memory_usage[str(element)] += 1

        if output_memory[i-1] == element:
            consecutive_output_memory_usage[str(element)] += 1


# sort output_memory_usage by keys
def sortDict(dictionary):
    elements = list(dictionary.keys())
    elements.sort(key=int)
    
    sorted_dictionary = {element: int(dictionary[element]) for element in elements}

    return sorted_dictionary

sorted_memory_usage = sortDict(memory_usage)
sorted_consecutive_output_memory_usage = sortDict(consecutive_output_memory_usage)

for element in sorted_memory_usage:
    optimized_memory_usage[element] = sorted_memory_usage[element] - sorted_consecutive_output_memory_usage[element]


if threshold:
    for element in optimized_memory_usage:
        if int(element) <= threshold:
            histogram_dict[element] = optimized_memory_usage[element]
        
        else:
            if histogram_dict.get('>' + str(threshold), None) is None:
                histogram_dict['>' + str(threshold)] = optimized_memory_usage[element]
            
            else:
                histogram_dict['>' + str(threshold)] += optimized_memory_usage[element]
            
            main_memory_total_memory_accesses += int(element) * optimized_memory_usage[element]

    ordered_histogram_dict['<1'] = histogram_dict.pop('0')
    ordered_histogram_dict.update(histogram_dict)


max_operator_memory = max([int(key) for key in optimized_memory_usage.keys()])

print('Insights:')
print('Maximum Memory Size of any Operator: {} MB'.format(max_operator_memory))
print('Total Memory of Operators that have memory size > {} MB: {} MB'.format(threshold, main_memory_total_memory_accesses))

if ordered_histogram_dict:
    fig, ax = plt.subplots(figsize=(12, 12))

    num_keys = len(ordered_histogram_dict)
    keys = list(ordered_histogram_dict.keys())
    values = list(ordered_histogram_dict.values())
    plot_keys = []
    plot_values = []

    # do not plot value when value percent < 1%
    del_idx = []
    for i, value in enumerate(values):
        if (value * 100.0) / sum(values) < 1.0:
            del_idx.append(i)
    
    for j in range(num_keys):
        if j not in del_idx:
            plot_keys.append(keys[j])
            plot_values.append(values[j])

    explode = [0] * len(plot_keys)
    explode[-1] = 0.1
    
    _, _, pcts = ax.pie(
        plot_values,
        explode=explode,
        labels=[key + ' MB' for key in plot_keys],
        autopct='%1d%%',
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        startangle=90)
    
    plt.setp(pcts, size=14, color='white', fontweight='bold')

    # set title
    fig.suptitle('{}\n'.format(model_name), fontweight='bold')

    ax.set_title('Should Weights + Output of an Operator\nbe stored in Main Memory ' + 
                 'during single inference?\n\nIf memory size of the Operator > {} MB\n'.format(threshold) + 
                 '(on-chip memory) with no NPU or SLC cache\n')

    plt.tight_layout()

    fig.savefig(os.path.join(save_directory, '_'.join(model_name.lower().split(' ')) + '_' + str(threshold) + 'mb_' + 'plot.png'))

    plt.show()
