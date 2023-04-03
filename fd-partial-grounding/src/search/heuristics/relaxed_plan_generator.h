#ifndef HEURISTICS_RELAXED_PLAN_GENERATOR_H
#define HEURISTICS_RELAXED_PLAN_GENERATOR_H

#include "relaxation_heuristic.h"

#include "../algorithms/priority_queues.h"

#include <vector>

namespace relaxed_plan_generator {

using relaxation_heuristic::PropID;
using relaxation_heuristic::OpID;

using relaxation_heuristic::NO_OP;

using relaxation_heuristic::Proposition;
using relaxation_heuristic::UnaryOperator;

enum PlanType {SINGLE_PLAN, ALL_PLANS};

class RelaxedPlanGenerator : public relaxation_heuristic::RelaxationHeuristic {
    // Relaxed plans are represented as a set of operators implemented
    // as a bit vector.
    typedef std::vector<bool> RelaxedPlan;
    RelaxedPlan relaxed_plan;

    // for every fact the list of best supporters
    std::vector<std::vector<OpID>> reached_by_actions;

    priority_queues::AdaptiveQueue<PropID> queue;

    bool save_relaxed_plan;
    bool save_relaxed_facts;
    std::string plan_filename;
    std::string facts_filename;
    PlanType plan_type;

    void setup_exploration_queue();
    void setup_exploration_queue_state(const State &state);
    void relaxed_exploration();

    void mark_preferred_operators_and_relaxed_plan(
            const State &state, PropID goal);

    void save_relaxed_plan_info_and_terminate(const State &state) const;

    void enqueue_if_necessary(PropID prop_id, int cost, OpID op_id) {
        assert(cost >= 0);
        Proposition *prop = get_proposition(prop_id);
        if (plan_type == ALL_PLANS && prop->cost == cost) {
            reached_by_actions[prop_id].push_back(op_id);
        }
        if (prop->cost == -1 || prop->cost > cost) {
            prop->cost = cost;
            prop->reached_by = op_id;
            if (plan_type == ALL_PLANS){
                reached_by_actions[prop_id] = std::vector<OpID>(1, op_id);
            }
            queue.push(cost, prop_id);
        }
        assert(prop->cost != -1 && prop->cost <= cost);
    }
protected:
    virtual int compute_heuristic(const State &ancestor_state);
public:
    RelaxedPlanGenerator(const options::Options &options);
    virtual ~RelaxedPlanGenerator() = default;
};
}

#endif
