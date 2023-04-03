#! /usr/bin/env python3

import itertools
import math
import os
from pathlib import Path
import subprocess

from lab.reports import Attribute, geometric_mean, arithmetic_mean

from downward.experiment import FastDownwardExperiment

from downward.reports.absolute import AbsoluteReport
from downward.reports.scatter import ScatterPlotReport

import common

DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

exp = FastDownwardExperiment()

domains = ["agricola", "blocks", "depots", "hiking", "satellite", "tpp", "zenotravel"]

exp.add_fetcher("data/2023-03-09-all-static-eval")

def rename_config(run):
    run["algorithm"] = run["algorithm"] "-inc"
    run["id"][0] = run["id"][0] + "-inc"
    return run


for domain in domains:
    exp.add_fetcher(f'data/2023-03-09-{domain}-fasttext-eval', merge=True)
    exp.add_fetcher(f'data/2023-03-09-{domain}-fasttext-opt-eval', merge=True)
    exp.add_fetcher(f'data/2023-03-09-{domain}-ratio-static-eval', merge=True)



exp.add_report(AbsoluteReport(attributes=common.ATTRIBUTES), name=SCRIPT_NAME)


RUNTIME_PLOT_CONFIG_PAIRS = [
    #["lpmm-b50k-sbmiasm-exactnocache", "lpmm-b50k-compliantsbmiasm-exact"],
    #["lpmm-gamerpdbs-addops", "lpmm-gamerpdbs-compliant-recursive"],
]


PNG_OR_TEX='tex'
for c1, c2 in RUNTIME_PLOT_CONFIG_PAIRS:
    report_name=f'{exp.name}-plot-searchtime-{c1}-vs-{c2}.{PNG_OR_TEX}'
    report_file=Path(exp.eval_dir) / f'{report_name}'
    exp.add_report(
        ScatterPlotReport(
            attributes='search_time',
            filter_algorithm=[c1, c2],
            # filter=[f.collect_non_factorable_tasks,f.filter_interesting_tasks],
            #get_category=f.is_strongly_compliant_task,
            format=PNG_OR_TEX,
            show_missing=False,
        ),
        name=report_name,
        outfile=report_file,
    )


exp.run_steps()
