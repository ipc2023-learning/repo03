from aleph_background import PredictionType
import os

# def run_yap_script():

#         proc = subprocess.Popen(['bash', yap_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         try:
#             output, error_output = proc.communicate(timeout=self.time_limit) # Timeout in seconds

#             aleph_output = output.decode()

#             if '[theory]' not in aleph_output:
#                 print (aleph_output, error_output.decode())

#             aleph_output = aleph_output[aleph_output.index('[theory]'):]

#             theory = aleph_output[9:aleph_output.index('[Training set performance]')].strip()

#             accuracy = float(self.regex_accuracy.search(aleph_output).group(1))

#             if theory:
#                 rules = []
#                 rules_text = theory.split('[Rule ')[1:]
#                 for r in rules_text:
#                     r = r.replace('\n', ' ')
#                     match = self.regex_rule.search(r)
#                     rules.append((match[1], match[2], match[3].strip()))

#                 self.saved_rules[instance].update(rules)

#                 match = self.regex_summary.search(aleph_output)
#                 true_positives, false_positives, false_negatives, true_negatives = int(match[1]), int(match[2]), int(match[3]), int(match[4])
#                 precision = true_positives/(true_positives + false_positives)
#                 recall = true_positives/(true_positives + false_negatives)
#                 fvalue = 2*precision*recall/(precision + recall)
#                 print (f"Aleph terminated with theory for {instance}: rules={len(rules)}, accuracy={accuracy}, precision={precision}, recall={recall}, fvalue={fvalue}, confusionmatrix = {true_positives} {false_positives} {false_negatives} {true_negatives}")
#                 print('\n'.join([r[2] for r in rules]))
#                 return 1-fvalue
#             else:
#                 print (f"Aleph terminated with no theory for {instance}")
#                 return 10
#         except subprocess.CalledProcessError:
#             print (f"WARNING: Aleph call failed")
#             print("Output: ", output.decode())

#             return 10000000

#         except subprocess.TimeoutExpired as timeErr:
#             print (f"Aleph terminated with timeout for {instance}")

#             output = timeErr.stdout
#             #error_output = timeErr.stderr


#             aleph_output = output.decode()

#             proc.kill()

#             rules = []
#             try:
#                 while True:
#                     aleph_output = aleph_output[aleph_output.index('[best clause]') + len('[best clause]'):]
#                     relevant_text = aleph_output[:aleph_output.index(']')+1].replace('\n','')
#                     # print(relevant_text)
#                     match = self.regex_rule_timeout.search(relevant_text)
#                     # print (relevant_text, match)
#                     if int(match[2]) > 1: # Skip rules without at least 2 positive examples
#                         rules.append((match[2], match[3], match[1].strip()))

#             except:
#                  pass # No more best clause elements

#             self.saved_rules[instance].update(rules)


#             print('\n'.join([r[2] for r in rules]))
#             # [best clause]
#             # class(A,B) :-
#             # 'ini:on'(A,C,B), 'goal:on'(C,D,B), 'ini:clear'(A,B), 'ini:on'(E,D,B).
#             # [pos cover = 22 neg cover = 0] [novelty] [0.0421258]


#             # print(output.decode())
#             # print(error_output.decode())

#             return 10



def get_aleph_parameters_and_command(prediction_type, extra_parameters):
    if prediction_type == PredictionType.class_probability:
        aleph_parameters = {'clauselength' : '10',
                     'lookahead' : '1',
                     'evalfn' : 'entropy',
                     'mingain' : '0.01',
                     'prune_tree' : 'false',
                     'confidence' : '0.001'}
        aleph_command = 'induce_tree'

    else:
        # learn rules with very few positive examples. We expect that these rules will be
        # double checked with respect to test instances and then their use is validated as
        # useful with the SMAC optimization

        aleph_parameters = {'clauselength' : '8',
                            'minacc' : '0.7',
                            'check_useless' : 'true',
                            'verbosity' : '0',
                            'minpos' : '10'}

        if  prediction_type == PredictionType.bad_actions:
            aleph_parameters['noise'] = 0

        aleph_parameters.update(extra_parameters)

        aleph_command = 'induce'
        if 'aleph_command' in aleph_parameters:
            aleph_command = aleph_parameters['aleph_command']

            aleph_parameters.pop('aleph_command')

    return aleph_parameters, aleph_command


def write_yap_file_internal(filename_path, YAP_PATH, EXAMPLES_PATH, HYPOTHESIS_FILE, aleph_parameters, aleph_command):

    yap_script_template = """
#!/bin/bash

cd "$(dirname "$0")"

YAP_PATH={YAP_PATH}
EXAMPLES_FILE={EXAMPLES_PATH}
HYPOTHESIS_FILE={HYPOTHESIS_FILE}

$YAP_PATH <<EOF
[aleph].

read_all('$EXAMPLES_FILE').

{ALEPH_CONFIGURATION}

write_rules('$HYPOTHESIS_FILE').
EOF
"""
    def yap_line (line):
        return f" {line}."

    def yap_set_line (a, b):
        return yap_line(f"set({a},{b})")

    ALEPH_CONFIGURATION = "\n".join([yap_set_line(str(a), str(b)) for a, b in aleph_parameters.items()] + [yap_line(aleph_command)])

    yap_content = yap_script_template.format(**locals())
    f = open(filename_path, 'w')
    f.write(yap_content)

    os.chmod(filename_path, 0o744)

    f.close()



def write_yap_file (filename_path, YAP_PATH, EXAMPLES_PATH, HYPOTHESIS_FILE, prediction_type, extra_parameters):
    aleph_parameters, aleph_command = get_aleph_parameters_and_command(prediction_type, extra_parameters)
    write_yap_file_internal (filename_path, YAP_PATH, EXAMPLES_PATH, HYPOTHESIS_FILE, aleph_parameters, aleph_command)
