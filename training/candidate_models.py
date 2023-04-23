from collections import defaultdict
import os
import shutil


class CandidateModels:
    def __init__(self):
        self.sk_models_per_action_schema = defaultdict(lambda : ['none'])
        self.good_rules = []
        self.bad_rules = []

    def is_using_model(self, config):
        return all ([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema]) and \
            any  ([config[f'model_{aschema}'] != 'none' for aschema in self.sk_models_per_action_schema])

    def get_unique_model_name(self, config):
        assert all([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema])
        prefix = lambda x : "sk" if x.startswith(PREFIX_SK_MODELS) else ("a" if x.endswith(SUFFIX_ALEPH_MODELS) else "")
        return "-".join([prefix(config[f'model_{aschema}']) + str(opts.index(config[f'model_{aschema}'])) for aschema, opts in self.sk_models_per_action_schema.items()])

    def load_sk_folder(self, sk_folder):
        self.sk_folder = sk_folder

        sk_models = [name for name in os.listdir(sk_folder) if name.startswith(PREFIX_SK_MODELS)]

        for model in sk_models:
            for n in os.listdir(os.path.join(sk_folder, model)):
                if n == 'relevant_rules':
                    continue
                self.sk_models_per_action_schema[n[:-6]].append(model)

    def load_aleph_folder(self, aleph_folder):
        self.aleph_folder = aleph_folder

        aleph_model_filenames = [name for name in os.listdir(aleph_folder) if name.endswith(SUFFIX_ALEPH_MODELS)]
        for model_filename in aleph_model_filenames:
            with open(os.path.join(aleph_folder, model_filename)) as model_file:
                if 'class_probability' in model_filename:
                    for line in model_file.readlines():
                        schema = line.split(":-")[ 0].strip()
                        self.sk_models_per_action_schema [schema].append(model_filename)
                elif 'good_rules' in model_filename:
                    self.good_rules += model_file.readlines()
                elif 'bad_rules' in model_filename:
                    self.bad_rules += model_file.readlines()
                else:
                    print (f"Warning: ignoring file of unknown type: {model_filename}")


    def copy_model_to_folder(self, config, target_dir, symlink=False ):
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        collected_relevant_rules = []
        collected_aleph_models = []
        for aschema in self.sk_models_per_action_schema:
            if config[f'model_{aschema}'].startswith(PREFIX_SK_MODELS):
                model_file = os.path.join(self.sk_folder, config[f'model_{aschema}'], aschema + ".model")
                target_file = os.path.join(target_dir, aschema + ".model")
                assert os.path.exists(model_file)
                if symlink:
                    os.symlink(model_file, target_file)
                else:
                    shutil.copy(model_file, target_file)

                with open(os.path.join(self.sk_folder, config[f'model_{aschema}'], 'relevant_rules')) as rfile:
                    for line in rfile:
                        if line.startswith (aschema + " ("):
                            collected_relevant_rules.append(line.strip())
            elif config[f'model_{aschema}'].endswith(SUFFIX_ALEPH_MODELS):
                with open(os.path.join(self.aleph_folder, config[f'model_{aschema}'])) as probability_model:
                    for line in probability_model.readlines():
                        schema = line.split(":-")[0].strip()
                        if schema == aschema:
                            collected_aleph_models.append(line)

            else:
                assert config[f'model_{aschema}'] == 'none'

        if collected_relevant_rules:
            with open(os.path.join(target_dir, 'relevant_rules'), 'w') as f:
                f.write('\n'.join(collected_relevant_rules))

        if collected_aleph_models:
            with open(os.path.join(target_dir, 'probability_class.rules'), 'w') as f:
                f.write('\n'.join(collected_aleph_models))

        selected_bad_rules = [r for i, r in enumerate(self.bad_rules) if [config[f"bad{i}"]]]
        if selected_bad_rules:
            with open(os.path.join(target_dir, 'bad_rules.rules'), 'w') as f:
                f.write('\n'.join(selected_bad_rules))

        selected_good_rules = [r for i, r in enumerate(self.good_rules) if [config[f"good{i}"]]]
        if selected_good_rules:
            with open(os.path.join(target_dir, 'good_rules.rules'), 'w') as f:
                f.write('\n'.join(selected_good_rules))