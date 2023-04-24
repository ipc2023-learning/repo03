#ifndef HEURISTICS_OPTIMAL_PLANS_HEURISTIC_H
#define HEURISTICS_OPTIMAL_PLANS_HEURISTIC_H


#include "additive_heuristic.h"

#include <vector>

#include "../symbolic/sym_controller.h"
#include "../symbolic/sym_enums.h"
#include "../search_engine.h"
#include <map>
#include <memory>
#include <iostream>

#include <fstream>

namespace symbolic {
    class SymStateSpaceManager;
    class SymSearch;
    class SymSolution;

}

namespace optimal_plans_heuristic {

using Proposition = relaxation_heuristic::Proposition;
using UnaryOperator = relaxation_heuristic::UnaryOperator;

    class OptimalPlansHeuristic : public additive_heuristic::AdditiveHeuristic, public symbolic::SymController {
    // Relaxed plans are represented as a set of operators implemented
    // as a bit vector.
    typedef std::vector<bool> RelaxedPlan;
    RelaxedPlan relaxed_plan;
    void mark_preferred_operators_and_relaxed_plan(
        const State &state, Proposition *goal);
protected:

    // Symbolic manager to perform bdd operations
    std::shared_ptr<symbolic::SymStateSpaceManager> mgr;

    std::unique_ptr<symbolic::SymSearch> search;

    const bool store_operators_in_optimal_plan;
    const bool store_relaxed_plan;
    std::map <const GlobalOperator *, BDD>  operators_in_optimal_plan;

    BDD all_states_in_optimal_plans;

        std::ofstream outfile_states;



    virtual int compute_heuristic(const GlobalState &global_state);

        virtual void new_solution(const symbolic::SymSolution & sol) override;
public:
    OptimalPlansHeuristic(const options::Options &options);
    ~OptimalPlansHeuristic();
};
}

#endif
