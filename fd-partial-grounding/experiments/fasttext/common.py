


DOMAINS = {
    "agricola":   ["agricola-large", "agricola-evaluation"],
    "blocks":     ["blocks-large", "blocksworld-evaluation"],
    "caldera":    ["caldera-large", "caldera-evaluation"],
    "depots":     ["depots-large", "depots-new-evaluation"],
    "hiking":     ["hiking-evaluation"], # IPC instances also?
    "satellite":  ["satellite-large", "satellite-evaluation"],
    "tpp":        ["tpp-large", "tpp-evaluation"],
    "zenotravel": ["zenotravel-evaluation"], # IPC instances also?
}

CLASSIFIERS = {
    "agricola":      "agricola",
    "blocks":        "blocksworld",
    "depots":        "depots-new",
    "hiking":        "hiking",
    "satellite":     "satellite",
    "tpp":           "tpp",
    "zenotravel":    "zenotravel",
}

ATTRIBUTES = [
        "cost",
        "coverage",
        "error",
        "evaluations",
        "expansions",
        "generated",
        "memory",
        "planner_memory",
        "planner_time",
        "run_dir",
        "search_time",
        "total_time",

        "num_ground_actions",
        "powerlifted_time",
        "grounding_queue_pushes",
        "translator_time",
]
