#ifndef SEARCH_HEURISTICS_USEFUL_FACTS_WRITER_H_
#define SEARCH_HEURISTICS_USEFUL_FACTS_WRITER_H_

#include "datalog_transformation_options.h"
#include "hmax_heuristic.h"

#include "../task.h"

class UsefulFactsWriter : public HMaxHeuristic{

    std::string useful_facts_filename;

public:
    UsefulFactsWriter(const Task &task) : UsefulFactsWriter(task, DatalogTransformationOptions()){};

    UsefulFactsWriter(const Task &task,
                      DatalogTransformationOptions opts,
                      std::string useful_facts_filename = "");

    int compute_heuristic(const DBState &s, const Task &task) override;
};

#endif //SEARCH_HEURISTICS_USEFUL_FACTS_WRITER_H_
