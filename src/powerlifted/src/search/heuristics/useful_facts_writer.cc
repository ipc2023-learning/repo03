#include "useful_facts_writer.h"

#include "utils.h"

#include <fstream>

using namespace std;

UsefulFactsWriter::UsefulFactsWriter(const Task &task,
                                     DatalogTransformationOptions opts,
                                     string useful_facts_filename) :
        HMaxHeuristic(task, opts),
        useful_facts_filename(useful_facts_filename) {}


static void _print_useful_atoms(const Task &task,
                                const vector<vector<GroundAtom>> &useful_atoms,
                                std::ostream &ostream,
                                bool single_line) {
    // TODO might be an option to discard goal facts
    for (size_t pred_idx = 0; pred_idx < useful_atoms.size(); ++pred_idx) {
        if (task.predicates[pred_idx].isStaticPredicate()) {
            continue;
        }
        string relation_name = task.predicates[pred_idx].get_name();
        for (const GroundAtom &atom : useful_atoms[pred_idx]) {
            ostream << relation_name << "(";
            if (atom.size() > 0){
                // there are 0-ary predicates
                ostream << task.objects[atom[0]].get_name();
            }
            for (int i = 1; i < static_cast<int>(atom.size()); ++i) {
                ostream << "," << task.objects[atom[i]].get_name();
            }
            if (single_line){
                ostream << "), ";
            } else {
                ostream << ")" << endl;
            }
        }
    }
    if (single_line) {
        ostream << endl;
    }
}

int UsefulFactsWriter::compute_heuristic(const DBState &s, const Task &task) {
    int h = HMaxHeuristic::compute_heuristic(s, task);

    if (h != std::numeric_limits<int>::max()) {
        if (useful_facts_filename != "FilePathUndefined") {
            ofstream useful_facts_file(useful_facts_filename);
            _print_useful_atoms(task, useful_atoms, useful_facts_file, false);
            useful_facts_file.close();
        } else {
            _print_useful_atoms(task, useful_atoms, cout, true);
        }
    }
    // print useful facts for the initial state and terminate
    exit(0);
}
