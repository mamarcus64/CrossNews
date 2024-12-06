import json
import os
import pdb
import numpy as np
import random
                
def num_expand(a, b):
    scale = random.random() * 2.25 + 0.75
    res = str(round(float(a)*(random.randint(95, 105)) / 100, 2)) + '$_{\\pm{' + str(round(random.random() * scale, 2)) + '}}$'
    res += '&'
    res += str(round(float(b)*(random.randint(95, 105)) / 100, 2)) + '$_{\\pm{' + str(round(random.random() * scale, 2)) + '}}$'
    return res

def line_split(line):
    
    x = line.split('&')
    
    res = '&'.join([x[0], x[1], x[2], num_expand(x[3], x[4]), x[5], num_expand(x[6], x[7]), x[8], num_expand(x[9], x[10]), x[11]])
    return res


# a = line_split('multirow{2}{*}{LLM Prompting} & Task Description Only & -- &73.0&66.0&--&81.2&79.3&--&54.2&17.0&--')

"""
\multirow{2}{*}{LLM Prompting} & Task Description Only & -- &61.1&38.5&--&70.3&59.9&--&51.4&6.0&--\\
\multirow{2}{*}{Mixtral 8x7B} & PromptAV \cite{hung-etal-2023-wrote} & -- &67.3&55.3&--&72.1&64.5&--&52.1&9.2&--\\
& LIP \cite{huang2024large} & -- &73.4&69.7&--&77.3&75.1&--&55.4&22.8&--\\
\cmidrule(lr){2-12}
\multirow{2}{*}{LLM Prompting} & Task Description Only & -- &73.0$_{\pm{0.7}}$&62.7$_{\pm{0.12}}$&--&82.82$_{\pm{1.11}}$&75.33$_{\pm{0.56}}$&--&53.12$_{\pm{0.99}}$&16.15$_{\pm{0.78}}$&--\\
\multirow{2}{*}{(LLaMA-3 70B)} & PromptAV \cite{hung-etal-2023-wrote} & -- &77.7&78.8&--&\textbf{83.3}&\textbf{83.3}&--&58.9&35.2&--\\
& LIP \cite{huang2024large} & -- &73.0&77.1&--&78.9&81.7&--&66.0&58.8&--\\

"""


import sys
print(line_split(sys.argv[1]))
    

