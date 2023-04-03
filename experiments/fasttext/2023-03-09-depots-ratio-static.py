#! /usr/bin/env python3

import itertools
import math
import os
from pathlib import Path
import subprocess

from lab.environments import TetralithEnvironment
from lab.reports import Attribute, geometric_mean, arithmetic_mean

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport
from downward.reports.compare import ComparativeReport
from downward.reports.scatter import ScatterPlotReport

import common

DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
REPO = os.path.dirname(os.path.dirname(DIR))
BENCHMARKS_DIR = os.environ["HTG_BENCHMARKS"]
REVISION = "57a7b7359cf3564f1d63d4ba73478731a353ebe9"

DOMAIN = SCRIPT_NAME[11:SCRIPT_NAME.index("-", 11)] # this requires proper script file names!

BUILD_OPTIONS = ["release"]
DRIVER_OPTIONS = ["--alias", "lama-first"]
INCREMENTAL_GROUNDING = ["--incremental-grounding", "--overall-time-limit", "90m", "--incremental-grounding-search-time-limit", str(30 * 60)]

RATIO_FILE = f"/proj/decoupled-search/ml_models/schema_ratios/{DOMAIN}.txt"

CONFIGS = [("ratio",   ["--translate-options", "--grounding-action-queue-ordering", "ratio",
                        "--termination-condition", "goal-relaxed-reachable",
                        "--action-schema-ratios", RATIO_FILE,
                        "--plan-ratios"]),

           ("ratio-inc", ["--translate-options", "--grounding-action-queue-ordering", "ratio",
                          "--action-schema-ratios", RATIO_FILE,
                          "--plan-ratios"]),
]

SUITE = sorted(common.DOMAINS[DOMAIN])

ENVIRONMENT = TetralithEnvironment(
    email="daniel.gnad@liu.se",
#    time_limit_per_task="24:00:00",
#    memory_per_cpu=None,
    extra_options="#SBATCH -A snic2022-22-1038",
    cpus_per_task=1, # hope that this actually limits every run to 1 core
)

exp = FastDownwardExperiment(
    environment=ENVIRONMENT,
)

for name, config in CONFIGS:
    if ("-inc" in name):
        exp.add_algorithm(name, REPO, REVISION, config, build_options=BUILD_OPTIONS, driver_options=DRIVER_OPTIONS + INCREMENTAL_GROUNDING)
    else:
        exp.add_algorithm(name, REPO, REVISION, config, build_options=BUILD_OPTIONS, driver_options=DRIVER_OPTIONS)


exp.add_suite(BENCHMARKS_DIR, SUITE)

exp.add_parser(exp.EXITCODE_PARSER)
#exp.add_parser(exp.TRANSLATOR_PARSER)
exp.add_parser(exp.ANYTIME_SEARCH_PARSER)
exp.add_parser(exp.PLANNER_PARSER)
exp.add_parser("incremental-parser.py")

exp.add_step('build', exp.build)
exp.add_step('start', exp.start_runs)
#exp.add_parse_again_step()
exp.add_fetcher(name='fetch')

exp.add_report(AbsoluteReport(attributes=common.ATTRIBUTES), name=SCRIPT_NAME)


RUNTIME_PLOT_CONFIG_PAIRS = [
#    [f"{BASE_REVISION}-lpmm-b50k-compliantsbmiasm-exactnocache", f"{REVISION}-lpmm-b50k-compliantsbmiasm-exact"],
#    [f"{BASE_REVISION}-lpmm-b50k-linearcggl-exactnocache",       f"{REVISION}-lpmm-b50k-linearcggl-exact"],
#    [f"{REVISION}-lpmm-b50k-sbmiasm-exactnocache",          f"{REVISION}-lpmm-b50k-sbmiasm-exact"],
#    [f"{REVISION}-lpmm-b50k-compliantsbmiasm-exact",        f"{REVISION}-lpmm-b50k-compliantsbmiasm-exactcompliant"],
#    [f"{REVISION}-lpmm-b50k-sbmiasm-exact",                 f"{REVISION}-lpmm-b50k-compliantsbmiasm-exact"],
]


PNG_OR_TEX='png'
for c1, c2 in RUNTIME_PLOT_CONFIG_PAIRS:
    report_name=f'{exp.name}-plot-searchtime-{c1}-vs-{c2}.{PNG_OR_TEX}'
    report_file=Path(exp.eval_dir) / f'{report_name}'
    exp.add_report(
        ScatterPlotReport(
            attributes='search_time',
            filter_algorithm=[c1, c2],
            # filter=filter_zero_nesting,
            # filter=[f.collect_non_factorable_tasks,f.filter_interesting_tasks],
            # get_category=lambda run1, run2: str(run1['ms_strongly_compliant'] if run1["algorithm"] == f"{REVISION}-lpmm-b50k-sbmiasm-exact" else run2["ms_strongly_compliant"]),
            format=PNG_OR_TEX,
            show_missing=False,
        ),
        name=report_name,
        outfile=report_file,
    )


exp.run_steps()
