#include "symbolic_search.h"


#include "../option_parser.h"
#include "../plugin.h"

#include "../symbolic/sym_search.h"
#include "../symbolic/sym_variables.h"
#include "../symbolic/sym_params_search.h"
#include "../symbolic/sym_state_space_manager.h"
#include "../symbolic/original_state_space.h"
#include "../symbolic/uniform_cost_search.h"
#include "../symbolic/bidirectional_search.h"

#include "../operator_cost_function.h"



using namespace std;
using namespace symbolic;
using namespace options;


namespace symbolic_search {

    SymbolicSearch::SymbolicSearch(const options::Options &opts) :
	SearchEngine(opts), SymController(opts),
	store_operators_in_optimal_plan(opts.get<bool> ("store_operators_in_optimal_plan")) {
    }

    SymbolicBidirectionalUniformCostSearch::SymbolicBidirectionalUniformCostSearch(const options::Options &opts) :
	SymbolicSearch(opts) {
    }

    void SymbolicBidirectionalUniformCostSearch::initialize() {
	mgr = make_shared<OriginalStateSpace> (vars.get(), mgrParams, OperatorCostFunction::get_cost_function(cost_type));
	auto fw_search = make_unique <UniformCostSearch> (this, searchParams);
	auto bw_search = make_unique <UniformCostSearch> (this, searchParams);
	fw_search->init(mgr, true, bw_search->getClosedShared());
	bw_search->init(mgr, false, fw_search->getClosedShared());

	search = make_unique<BidirectionalSearch> (this, searchParams, move(fw_search), move(bw_search));
    }


    SymbolicUniformCostSearch::SymbolicUniformCostSearch(const options::Options &opts, bool _fw) :
	SymbolicSearch(opts), fw(_fw) {
    }

    void SymbolicUniformCostSearch::initialize() {
	mgr = make_shared<OriginalStateSpace> (vars.get(), mgrParams, OperatorCostFunction::get_cost_function(cost_type));
        auto uni_search = make_unique <UniformCostSearch> (this, searchParams);
	if(fw) {
	    uni_search->init(mgr, true, nullptr);
	}else {
	    uni_search->init(mgr, false, nullptr);
	}

	search.reset(uni_search.release());

    }



    SearchStatus SymbolicSearch::step() {
	search->step();

	if(getLowerBound() < getUpperBound()){
	    return IN_PROGRESS;
	}else if (found_solution()) {
	    if(store_operators_in_optimal_plan) {
		ofstream outfile("good_operators");

		for (auto * op : operators_in_optimal_plan) {
		    outfile << op->get_name() << endl;
		}
	    }
	    return SOLVED;
	}else{
	    return FAILED;
	}
    }

    void SymbolicSearch::new_solution(const SymSolution &sol) {
	if (sol.getCost() < getUpperBound()) {
	    vector <const GlobalOperator *> plan;
	    sol.getPlan(plan);
	    set_plan(plan);
	}

	if(store_operators_in_optimal_plan) {
	    // if (cost_type != OperatorCost::ONE) {
	    //     cerr << "Unsupported option: store_operators_in_optimal_plan but not cost_type=1" << endl;
	    //     utils::exit_with(utils::ExitCode::UNSUPPORTED);
	    // }
	    if (sol.getCost() <= getUpperBound()) {
		if (sol.getCost() < getUpperBound()) {
		    operators_in_optimal_plan.clear();
		}
		sol.getOperatorsOptimalPlans(operators_in_optimal_plan);
	    }
	}



	SymController::new_solution(sol);
    }
}

static SearchEngine *_parse_bidirectional_ucs(OptionParser &parser) {
    parser.document_synopsis("Symbolic Bidirectional Uniform Cost Search", "");

    SearchEngine::add_options_to_parser(parser);
    SymVariables::add_options_to_parser(parser);
    SymParamsSearch::add_options_to_parser(parser, 30e3, 10e7);
    SymParamsMgr::add_options_to_parser(parser);
    parser.add_option<bool>("store_operators_in_optimal_plan", "store_operators_in_optimal_plan", "false");

    Options opts = parser.parse();

    symbolic_search::SymbolicSearch *engine = nullptr;
    if (!parser.dry_run()) {
        engine = new symbolic_search::SymbolicBidirectionalUniformCostSearch(opts);
    }

    return engine;
}

static SearchEngine *_parse_forward_ucs(OptionParser &parser) {
    parser.document_synopsis("Symbolic Bidirectional Uniform Cost Search", "");

    SearchEngine::add_options_to_parser(parser);
    SymVariables::add_options_to_parser(parser);
    SymParamsSearch::add_options_to_parser(parser, 30e3, 10e7);
    SymParamsMgr::add_options_to_parser(parser);
    parser.add_option<bool>("store_operators_in_optimal_plan", "store_operators_in_optimal_plan", "false");

    Options opts = parser.parse();

    symbolic_search::SymbolicSearch *engine = nullptr;
    if (!parser.dry_run()) {
        engine = new symbolic_search::SymbolicUniformCostSearch(opts, true);
    }

    return engine;
}

static SearchEngine *_parse_backward_ucs(OptionParser &parser) {
    parser.document_synopsis("Symbolic Bidirectional Uniform Cost Search", "");

    SearchEngine::add_options_to_parser(parser);
    SymVariables::add_options_to_parser(parser);
    SymParamsSearch::add_options_to_parser(parser, 30e3, 10e7);
    SymParamsMgr::add_options_to_parser(parser);
    parser.add_option<bool>("store_operators_in_optimal_plan", "store_operators_in_optimal_plan", "false");

    Options opts = parser.parse();

    symbolic_search::SymbolicSearch *engine = nullptr;
    if (!parser.dry_run()) {
        engine = new symbolic_search::SymbolicUniformCostSearch(opts, false);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_bd("sbd", _parse_bidirectional_ucs);
static Plugin<SearchEngine> _plugin_fw("sfw", _parse_forward_ucs);
static Plugin<SearchEngine> _plugin_bw("sbw", _parse_backward_ucs);
