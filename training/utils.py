import os
import tarfile
import shutil

class SaveModel:
    def __init__(self, knowledge_file, keep_copies=True):
        self.knowledge_file = knowledge_file
        self.keep_copies = 1 if keep_copies else 0

    def save(self, source_dir):
        if not self.knowledge_file:
            return

        with tarfile.open(self.knowledge_file + '.tmp', "w:gz", dereference=True) as tar:
            for f in os.listdir(source_dir):
                tar.add(os.path.join(source_dir, f), arcname=f)

        knowledge_filename = self.knowledge_file
        if self.keep_copies:
            knowledge_filename += f'.{self.keep_copies}'
            self.keep_copies += 1

        shutil.move(self.knowledge_file + '.tmp', knowledge_filename)
