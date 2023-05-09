from .dataset import SkipGramDataset
from .preprocess import Tokenizer
from glob import glob
import os
from tqdm import tqdm


class COHADataset(SkipGramDataset):

    def __init__(self, args, examples_path=None, dict_path=None):
        SkipGramDataset.__init__(self, args)
        self.name = 'COHA'
        self.queries = ['man', 'woman', 'doctor', 'nurse', 'asian', 'white']
        self.dataset_dir = args.dataset_dir

        if examples_path is not None and dict_path is not None:
            print('[INFO] Loading examples')
            self.load(examples_path, dict_path)
        else:
            self.tokenizer = Tokenizer(args)
            # Set self.files to a list of tokenized data!
            self.files = self.tokenize_files()
            # Generates examples in window size - e.g. (center_idx, context_idx)
            self.generate_examples_serial()
            # Save dataset files - this tokenization and example generation can take awhile with a lot of data
            print('Saving example and dictionary files')
            self.save('training_examples.pth', 'dictionary.pth')

        print(f'There are {len(self.dictionary)} tokens and {len(self.examples)} examples.')

    def load_files(self):
        """ Requires by SkipGramDataset to generate examples - must be tokenized files """
        return self.files

    def tokenize_files(self):
        # read in from a file or wherever your data is kept
        #raw_data = ["this is document_1", "this is document_2", ..., "this is document_n"]
        print('Tokenizing files')
        tokenized_data = []
        files = glob(os.path.join(self.dataset_dir, "**/*.txt"))
        # Start with just 2000s files
        files = glob(os.path.join(self.dataset_dir, "2000s/*.txt"))
        for file in tqdm(files):
            with open(file, 'r') as f:
                text = f.read()
                tokenized_data.append(self.tokenizer.tokenize_doc(text))

        return tokenized_data