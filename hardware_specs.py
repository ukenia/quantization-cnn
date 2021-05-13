# purpose: getting CPU and GPU specs from the system for comparison
# date: 4th May, 2021
# comments: output path for CPU stats should be json file and for GPU should be txt file

# import required libraries

import argparse
import cpuinfo
import json
import os

# get arguments from command line

parser = argparse.ArgumentParser()

parser.add_argument('--outpath', type=str, help='Path to the output file for specs')
parser.add_argument('--cpu', type=int, default=1,  help='Is the model to be quantized')

arguments = parser.parse_args()
output_path = arguments.outpath
cpu = arguments.cpu

def cpu_stats(output_path):
	# get cpu facts as a json
	with open(output_path,"w") as file:
		json.dump(cpuinfo.get_cpu_info_json(),file)

	print("CPU stats saved at: "+output_path)
	
def gpu_stats(output_path):
	# get gpu stats as a txt file
	os.system('nvidia-smi >> '+output_path)

	print("GPU stats saved at: "+output_path)

if __name__ == '__main__':
	if cpu==1:
		cpu_stats(output_path)
	else:
		gpu_stats(output_path)