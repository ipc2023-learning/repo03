#include "optimal_plans_heuristic.h"

#include "../global_state.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_tools.h"

#include "../symbolic/sym_search.h"
#include "../symbolic/sym_variables.h"
#include "../symbolic/sym_params_search.h"
#include "../symbolic/sym_state_space_manager.h"
#include "../symbolic/original_state_space.h"
#include "../symbolic/uniform_cost_search.h"
#include "../symbolic/bidirectional_search.h"
#include "../successor_generator.h"

#include <cassert>
#include <iterator>

using namespace std;
using namespace symbolic;

namespace optimal_plans_heuristic {
// construction and destruction
OptimalPlansHeuristic::OptimalPlansHeuristic(const Options &opts)
    : AdditiveHeuristic(opts), SymController(opts),
      relaxed_plan(task_proxy.get_operators().size(), false),
      store_operators_in_optimal_plan(opts.get<bool> ("store_operators_in_optimal_plan")),
      store_relaxed_plan(opts.get<bool> ("store_relaxed_plan")),
      outfile_states("good_operators_per_state") {

    cout << "Initializing optimal plans heuristic..." << endl;

    mgr = make_shared<OriginalStateSpace> (vars.get(), mgrParams, OperatorCostFunction::get_cost_function(cost_type));
    auto fw_search = make_unique <UniformCostSearch> (this, searchParams);
    auto bw_search = make_unique <UniformCostSearch> (this, searchParams);
    fw_search->init(mgr, true, bw_search->getClosedShared());
    bw_search->init(mgr, false, fw_search->getClosedShared());

    search = make_unique<BidirectionalSearch> (this, searchParams, move(fw_search), move(bw_search));


    while(getLowerBound() < getUpperBound()){
    	search->step();
    }

    if(store_operators_in_optimal_plan) {
        ofstream outfile_good_operators("good_operators");

        for (auto & op : operators_in_optimal_plan) {
            outfile_good_operators << op.first->get_name() << endl;
        }
    }

    all_states_in_optimal_plans = vars->zeroBDD();

    for (auto & op : operators_in_optimal_plan) {
        all_states_in_optimal_plans += op.second;
    }

    cout << "We have " << vars->numStates(all_states_in_optimal_plans) <<
        " states in the optimal plan, in a BDD with "
         << all_states_in_optimal_plans.nodeCount() << " nodes."  << endl;
    // assert (mgr->getVars()->isIn(g_initial_state, all_states_in_optimal_plans));

    outfile_states << "operators: \n";
    for (size_t i = 0; i < g_operators.size(); ++ i){
        outfile_states << g_operators[i].get_name() << '\n';
    }

    for (size_t i = 0; i < g_fact_names.size(); ++ i){
        outfile_states << "variable " << i << '\n';
        for (size_t j = 0; j < g_fact_names[i].size(); ++ j){
            outfile_states << g_fact_names[i][j] << '\n';
        }

    }



}



    void OptimalPlansHeuristic::new_solution(const SymSolution &sol) {
	if (sol.getCost() < getUpperBound()) {
	    vector <const GlobalOperator *> plan;
	    sol.getPlan(plan);
	    // set_plan(plan);
	}

	    if (cost_type != OperatorCost::ONE) {
	        cerr << "Unsupported option: store_operators_in_optimal_plan but not cost_type=1" << endl;
	        utils::exit_with(utils::ExitCode::UNSUPPORTED);
	    }
	    if (sol.getCost() <= getUpperBound()) {
		if (sol.getCost() < getUpperBound()) {
		    operators_in_optimal_plan.clear();
		}
		sol.getOperatorsOptimalPlans(operators_in_optimal_plan);
	    }



	SymController::new_solution(sol);
    }

OptimalPlansHeuristic::~OptimalPlansHeuristic() {
}

void OptimalPlansHeuristic::mark_preferred_operators_and_relaxed_plan(
    const State &state, Proposition *goal) {
    if (!goal->marked) { // Only consider each subgoal once.
        goal->marked = true;
        UnaryOperator *unary_op = goal->reached_by;
        if (unary_op) { // We have not yet chained back to a start node.
            for (size_t i = 0; i < unary_op->precondition.size(); ++i)
                mark_preferred_operators_and_relaxed_plan(
                    state, unary_op->precondition[i]);
            int operator_no = unary_op->operator_no;
            if (operator_no != -1) {
                // This is not an axiom.
                relaxed_plan[operator_no] = true;

                if (unary_op->cost == unary_op->base_cost) {
                    // This test is implied by the next but cheaper,
                    // so we perform it to save work.
                    // If we had no 0-cost operators and axioms to worry
                    // about, it would also imply applicability.
                    OperatorProxy op = task_proxy.get_operators()[operator_no];
                    if (is_applicable(op, state))
                        set_preferred(op);
                }
            }
        }
    }
}

    int get_index (const GlobalOperator *op) {
        return (op - g_operators.data());
    }
int OptimalPlansHeuristic::compute_heuristic(const GlobalState &global_state) {
    if (test_goal(global_state)) {
        return 1; // Make sure that the goal states are expanded last
    }

    if (mgr->getVars()->isIn(global_state, all_states_in_optimal_plans)){
        outfile_states << "state: ";
        for (size_t v = 0; v < g_fact_names.size(); ++v) {
            outfile_states << global_state[v] << ' ';
            //outfile_states << g_fact_names[v][global_state[v]] << '\n';
        }

        // for (auto & op : operators_in_optimal_plan) {
        // }

        // outfile_states << "\napplicable operators: " << endl;

        vector<const GlobalOperator *> ops;
        g_successor_generator->generate_applicable_ops(global_state, ops);
        vector<int> good_op_ids, bad_op_ids;

        for (auto * op : ops) {
            if (operators_in_optimal_plan.count(op) &&
                mgr->getVars()->isIn(global_state, operators_in_optimal_plan.at(op))) {
                good_op_ids.push_back(get_index(op));
            } else {
                bad_op_ids.push_back(get_index(op));
            }
        }

        outfile_states << "\n+: ";
        for (int id : good_op_ids) {
            outfile_states << id << " ";
        }

        outfile_states << "\n-: ";
        for (int id : bad_op_ids) {
            outfile_states << id << " ";
        }

        if (store_relaxed_plan) {
            outfile_states << "\nrelaxed: ";
            //compute hff to print relaxed plan
            State state = convert_global_state(global_state);
            int h_add = compute_add_and_ff(state);
            assert (h_add != DEAD_END);

            // Collecting the relaxed plan also sets the preferred operators.
            for (size_t i = 0; i < goal_propositions.size(); ++i)
                mark_preferred_operators_and_relaxed_plan(state, goal_propositions[i]);

            // int h_ff = 0;
            for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
                if (relaxed_plan[op_no]) {
                    relaxed_plan[op_no] = false; // Clean up for next computation.
                    // h_ff += task_proxy.get_operators()[op_no].get_cost();
                    outfile_states << get_index(task_proxy.get_operators()[op_no].get_global_operator()) << " ";
                }
            }
        }
        outfile_states << endl;

        return 0;
    } else {
        return DEAD_END;
    }
}


static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("Optimal Plans heuristic.", "");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes");
    parser.document_property("preferred operators", "yes");

    Heuristic::add_options_to_parser(parser);

    SymVariables::add_options_to_parser(parser);
    SymParamsSearch::add_options_to_parser(parser, 30e3, 10e7);
    SymParamsMgr::add_options_to_parser(parser);
    parser.add_option<bool>("store_operators_in_optimal_plan", "store_operators_in_optimal_plan", "true");
    parser.add_option<bool>("store_relaxed_plan", "store_relaxed_plan", "true");

    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new OptimalPlansHeuristic(opts);
}

static Plugin<Heuristic> _plugin("optimal_plans_heuristic", _parse);
}
