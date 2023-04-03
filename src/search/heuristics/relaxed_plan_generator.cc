#include "relaxed_plan_generator.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cassert>
#include <fstream>
#include <iostream>

using namespace std;

namespace relaxed_plan_generator {
// construction and destruction
RelaxedPlanGenerator::RelaxedPlanGenerator(const Options &opts)
    : RelaxationHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false),
      save_relaxed_plan(opts.get<bool>("save_relaxed_plan")),
      save_relaxed_facts(opts.get<bool>("save_relaxed_facts")),
      plan_filename(opts.get<string>("plan_filename")),
      facts_filename(opts.get<string>("facts_filename")),
      plan_type(PlanType(opts.get<PlanType>("plan_type"))) {
    cout << "Initializing Relaxed Plan Generator..." << endl;

    if (plan_type == ALL_PLANS){
        reached_by_actions.resize(propositions.size());
    }
}

void RelaxedPlanGenerator::mark_preferred_operators_and_relaxed_plan(
        const State &state, PropID goal_id) {
    Proposition *goal = get_proposition(goal_id);
    if (!goal->marked) { // Only consider each subgoal once.
        goal->marked = true;
        if (plan_type == ALL_PLANS){
            for (OpID op_id : reached_by_actions[goal_id]){
                bool is_preferred = true;
                UnaryOperator *unary_op = get_operator(op_id);
                if (unary_op) { // We have not yet chained back to a start node.
                    for (PropID precond : get_preconditions(op_id)){
                        mark_preferred_operators_and_relaxed_plan(
                                state, precond);
                        if (get_proposition(precond)->reached_by != NO_OP){
                            is_preferred = false;
                        }
                    }
                    int operator_no = unary_op->operator_no;
                    if (is_preferred && operator_no != -1) {
                        // This is not an axiom.
                        relaxed_plan[operator_no] = true;
                    }
                }
            }
        } else {
            OpID op_id = goal->reached_by;
            if (op_id != NO_OP) { // We have not yet chained back to a start node.
                bool is_preferred = true;
                UnaryOperator *unary_op = get_operator(op_id);
                for (PropID precond : get_preconditions(op_id)){
                    mark_preferred_operators_and_relaxed_plan(
                            state, precond);
                    if (get_proposition(precond)->reached_by != NO_OP){
                        is_preferred = false;
                    }
                }
                int operator_no = unary_op->operator_no;
                if (is_preferred && operator_no != -1) {
                    // This is not an axiom.
                    relaxed_plan[operator_no] = true;
                }
            }
        }
    }
}

void RelaxedPlanGenerator::save_relaxed_plan_info_and_terminate(const State &/*state*/) const {
    ofstream plan_file;
    if (save_relaxed_plan){
        plan_file = ofstream(plan_filename);
    }
    ofstream facts_file;
    if (save_relaxed_facts){
        facts_file = ofstream(facts_filename);
    }

    // TODO we could properly sequence the relaxed plan
    // TODO two options for facts output:
    // (1) print only necessary facts achieved by plan;
    // (2) print all facts (done here.. though not correctly working for conditional effects)


    cout << "Relaxed plan:" << endl;

    int plan_cost = 0;
    OperatorsProxy operators = task_proxy.get_operators();
    for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
        if (relaxed_plan[op_no]) {
            cout << operators[op_no].get_name() << " (" << operators[op_no].get_cost() << ")" << endl;
            if (save_relaxed_plan){
                plan_file << "(" << operators[op_no].get_name() << ")" << endl;
            }
            if (save_relaxed_facts){
                for (EffectProxy eff : operators[op_no].get_effects()){
                    facts_file << eff.get_fact().get_name() << endl;
                }
            }
            plan_cost += task_proxy.get_operators()[op_no].get_cost();
        }
    }

    if (save_relaxed_plan){
        bool is_unit_cost = task_properties::is_unit_cost(task_proxy);
        plan_file << "; cost = " << plan_cost << " ("
                << (is_unit_cost ? "unit cost" : "general cost") << ")" << endl;
        plan_file.close();
        cout << "Terminating - relaxed plan successfully saved to " << plan_filename << endl;
    }
    if (save_relaxed_facts){
        facts_file.close();
        cout << "Terminating - relaxed facts successfully saved to " << facts_filename << endl;
    }

    exit_with(utils::ExitCode::SEARCH_UNSOLVED_INCOMPLETE);
}

// heuristic computation
void RelaxedPlanGenerator::setup_exploration_queue() {
    queue.clear();

    for (Proposition &prop : propositions) {
        prop.cost = -1;
        prop.marked = false;
    }

    for (vector<OpID> &ops : reached_by_actions){
        ops.clear();
    }

    // Deal with operators and axioms without preconditions.
    for (UnaryOperator &op : unary_operators) {
        op.unsatisfied_preconditions = op.num_preconditions;
        op.cost = op.base_cost; // will be increased by precondition costs

        if (op.unsatisfied_preconditions == 0)
            enqueue_if_necessary(op.effect, op.base_cost, get_op_id(op));
    }
}

void RelaxedPlanGenerator::setup_exploration_queue_state(const State &state) {
    for (FactProxy fact : state) {
        PropID init_prop = get_prop_id(fact);
        enqueue_if_necessary(init_prop, 0, NO_OP);
    }
}

void RelaxedPlanGenerator::relaxed_exploration() {
    int unsolved_goals = goal_propositions.size();
    while (!queue.empty()) {
        pair<int, PropID> top_pair = queue.pop();
        int distance = top_pair.first;
        PropID prop_id = top_pair.second;
        Proposition *prop = get_proposition(prop_id);
        int prop_cost = prop->cost;
        assert(prop_cost >= 0);
        assert(prop_cost <= distance);
        if (prop_cost < distance)
            continue;
        if (prop->is_goal && --unsolved_goals == 0)
            return;
        for (OpID op_id : precondition_of_pool.get_slice(
                 prop->precondition_of, prop->num_precondition_occurences)) {
            UnaryOperator *unary_op = get_operator(op_id);
            --unary_op->unsatisfied_preconditions;
            unary_op->cost = max(unary_op->cost,
                                 unary_op->base_cost + prop_cost);
            assert(unary_op->unsatisfied_preconditions >= 0);
            if (unary_op->unsatisfied_preconditions == 0)
                enqueue_if_necessary(unary_op->effect, unary_op->cost, op_id);
        }
    }
}

int RelaxedPlanGenerator::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    setup_exploration_queue();
    setup_exploration_queue_state(state);
    relaxed_exploration();

    int total_cost = 0;
    for (PropID goal_id : goal_propositions) {
        const Proposition *goal = get_proposition(goal_id);
        int goal_cost = goal->cost;
        if (goal_cost == -1) {
            // TODO print warning and terminate
            return DEAD_END;
        }
        total_cost = max(total_cost, goal_cost);
    }

    // Collecting the relaxed plan also sets the preferred operators.
    for (size_t i = 0; i < goal_propositions.size(); ++i)
        mark_preferred_operators_and_relaxed_plan(state, goal_propositions[i]);

    save_relaxed_plan_info_and_terminate(state);

    // this code is never reached

    return total_cost;
}

static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis("Relaxed plan generator", "");
    parser.document_language_support("action costs", "supported");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support(
        "axioms",
        "supported (in the sense that the planner won't complain -- "
        "handling of axioms might be very stupid "
        "and even render the heuristic unsafe)");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes for tasks without axioms");
    parser.document_property("preferred operators", "yes");
    
    parser.add_option<bool>("save_relaxed_plan",
            "If true, the heuristic will only evaluate once (on the initial"
            "state), save the relaxed plan to the file specified by the"
            "\"plan_filename\" option (default is relaxed_plan), and exit the"
            "planner.",
            "false");
    parser.add_option<bool>("save_relaxed_facts",
                "If true, the heuristic will only evaluate once (on the initial"
                "state), save the facts achieved along the relaxed plan to the "
                "file specified by the \"facts_filename\" option (default is "
                "relaxed_facts), and exit the planner.",
                "false");
    parser.add_option<string>("plan_filename",
            "If the save_relaxed_plan option is set to true, the generated "
            "relaxed plan will be saved to this file. Otherwise, this option "
            "has no effect.",
            "relaxed_plan");
    parser.add_option<string>("facts_filename",
                "If the save_relaxed_facts option is set to true, the facts "
                "achieved along the generated relaxed plan will be saved to "
                "this file. Otherwise, this option has no effect.",
                "relaxed_facts");

    vector<string> plan_types;
    plan_types.push_back("SINGLE_PLAN");
    plan_types.push_back("ALL_PLANS");

    parser.add_enum_option<PlanType>("plan_type",
                plan_types,
                "Specifies how the relaxed plan should be generated. SINGLE_PLAN "
                "is the default that computes a standard relaxed plan. ALL_PLANS"
                "is an approximation of the set of all relaxed plans.",
                "SINGLE_PLAN");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<RelaxedPlanGenerator>(opts);
}

static Plugin<Evaluator> _plugin("gen_relaxed_plan", _parse);
}
