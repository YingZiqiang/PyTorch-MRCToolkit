import pickle
import logging
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader


class BatchGenerator(object):
    def __init__(self):
        pass

    def build(self, vocab, instances,
              batch_size=32,
              training=False,
              max_context_len=500,
              max_question_len=30,
              use_char=True,
              max_word_len=30,
              additional_fields=None,
              feature_vocab=None,
              num_parallel_calls=0):
        """
        Build the batch generator, including build dataset and build dataloader
        """
        self.vocab = vocab
        self.instances = instances
        self.batch_size = batch_size
        self.training = training
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.use_char = use_char
        self.max_word_len = max_word_len
        self.additional_fields = additional_fields if additional_fields is not None else list()
        self.feature_vocab = feature_vocab if feature_vocab is not None else dict()
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls > 0 else multiprocessing.cpu_count() // 2
        if self.instances is None or len(self.instances) == 0:
            raise ValueError('empty instances!!')

        self.dataset = self._build_dataset_pipeline()
        self.dataloader = self._build_dataloader_pipeline()

    def save(self, file_path):
        """
        Save the attribute of BatchGenerator
        """
        logging.info("Saving BatchGenerator at {}".format(file_path))
        # pickle can't save generator and dataloader, so we skip those fields
        dataloader_tmp = self.dataloader
        self.generator, self.dataloader = None, None
        with open(file_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        self.dataloader = dataloader_tmp

    def load(self, file_path):
        """
        Load the saved file and rebuilt BatchGenerator
        """
        logging.info("Loading BatchGenerator at {}".format(file_path))
        with open(file_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.__dict__.update(vocab_data)
        # we don't save value of generator and dataloader, so get they here
        self.generator = None
        self.dataloader = self._build_dataloader_pipeline()

    def init(self):
        """
        Initialize the dataloader generator
        """
        self.generator = BatchGenerator._generator(self.dataloader)

    def next(self):
        """
        Get next batch data of dataloader
        """
        if self.generator is None:
            raise Exception('you must do init before do next.')
        return next(self.generator)

    def get_dataset_size(self):
        return len(self.dataset)

    def get_batch_size(self):
        return self.batch_size

    def get_raw_dataset(self):
        return self.instances

    @staticmethod
    def _generator(dataloader):
        for batch_data in dataloader:
            yield batch_data

    @staticmethod
    def _dynamic_padding(example, pad_len, pad_thing):
        example = (example + [pad_thing] * (pad_len - len(example)))[:pad_len]
        return example

    @staticmethod
    def _detect_input_type(instance, additional_fields=None):
        instance_keys = instance.keys()
        fields = ['context_tokens', 'question_tokens', 'answer_start', 'answer_end']
        try:
            for f in fields:
                assert f in instance_keys
        except Exception:
            raise ValueError('A instance should contain at least "context_tokens", "question_tokens", \
                             "answer_start", "answer_end" four fields!')

        if additional_fields is not None and isinstance(additional_fields, list):
            fields.extend(additional_fields)

        def get_type(value):
            if isinstance(value, float):
                return torch.float32
            elif isinstance(value, int):
                return torch.int64
            elif isinstance(value, str):
                return str
            elif isinstance(value, bool):
                return bool
            else:
                return None

        input_type = {}

        for field in fields:
            if instance[field] is None:
                if field not in ('answer_start', 'answer_end'):
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
                continue
            elif isinstance(instance[field], list):
                if len(instance[field]) == 0:
                    logging.warning('Data shape of field "%s" not detected! Skip this field.', field)
                    continue

                field_type = get_type(instance[field][0])
                if field_type is not None:
                    input_type[field] = field_type
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
            else:
                field_type = get_type(instance[field])
                if field_type is not None:
                    input_type[field] = field_type
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)

        return fields, input_type

    def _build_dataset_pipeline(self):
        # 1. Check the input-data type and filter invalid keys
        input_fields, input_type_dict = BatchGenerator._detect_input_type(self.instances[0], self.additional_fields)
        filtered_instances = [{k: instance[k] for k in input_fields} for instance in self.instances]

        # 3.3 other feature look-up table
        # if len(self.feature_vocab) > 0:
        #     feature_table = {}
        #     for feature_name, vocab in self.feature_vocab.items():
        #         feature_table[feature_name] = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab),
        #                                                                                 num_oov_buckets=1)

        # 4. Some preprocessing, including char extraction, lowercasing, length
        def transform_new_instance(instance):
            context_tokens = instance['context_tokens']
            question_tokens = instance['question_tokens']

            if self.use_char:
                def get_seq_char_ids(word_tokens):
                    result = []
                    for word in word_tokens:
                        word_char_ids = [self.vocab.get_char_idx(char) for char in word]
                        result.append(word_char_ids)
                    return result
                instance['context_char_ids'] = get_seq_char_ids(context_tokens)
                instance['question_char_ids'] = get_seq_char_ids(question_tokens)
                instance['context_word_len'] = [len(word) for word in context_tokens]
                instance['question_word_len'] = [len(word) for word in question_tokens]

            # if do_lowercasing, we do it in get_word_idx function
            instance['context_ids'] = [self.vocab.get_word_idx(token) for token in context_tokens]
            instance['question_ids'] = [self.vocab.get_word_idx(token) for token in question_tokens]
            instance['context_len'] = len(context_tokens)
            instance['question_len'] = len(question_tokens)
            # if len(self.feature_vocab) > 0:
            #     for field in self.additional_fields:
            #         for feature_name, table in feature_table.items():
            #             if field.endswith(feature_name):
            #                 instance[field] = tf.cast(table.lookup(instance[field]), tf.int32)
            #                 break

            for field, field_type in input_type_dict.items():
                if field_type == str:
                    del instance[field]

            return instance
        new_instances = [transform_new_instance(instance) for instance in filtered_instances]

        return MRCDataset(new_instances)

    def _build_dataloader_pipeline(self):
        word_pad_idx = self.vocab.get_word_pad_idx()
        if self.use_char:
            char_pad_idx = self.vocab.get_char_pad_idx()

        def mrc_collate(batch):
            result = {}
            for key in batch[0].keys():
                result[key] = []

            # 1. Handle the word level sequence data
            # 1.1 Get batch pad length
            pad_context_len = min(self.max_context_len, max([sample['context_len'] for sample in batch]))
            pad_question_len = min(self.max_question_len, max([sample['question_len'] for sample in batch]))

            # 1.2 Padding context and question
            for sample in batch:
                sample['context_ids'] = BatchGenerator._dynamic_padding(sample['context_ids'], pad_context_len, word_pad_idx)
                sample['question_ids'] = BatchGenerator._dynamic_padding(sample['question_ids'], pad_question_len, word_pad_idx)
                sample['context_len'] = min(sample['context_len'], pad_context_len)
                sample['question_len'] = min(sample['question_len'], pad_question_len)

            # 2. Handle the char level data
            if self.use_char:
                # 2.1 Padding sample `char ids` and `word len` to batch max length
                for sample in batch:
                    sample['context_char_ids'] = BatchGenerator._dynamic_padding(
                        sample['context_char_ids'], pad_context_len, [char_pad_idx])
                    sample['question_char_ids'] = BatchGenerator._dynamic_padding(
                        sample['question_char_ids'], pad_question_len, [char_pad_idx])
                    sample['context_word_len'] = BatchGenerator._dynamic_padding(
                        sample['context_word_len'], pad_context_len, 0)
                    sample['question_word_len'] = BatchGenerator._dynamic_padding(
                        sample['question_word_len'], pad_question_len, 0)

                # 2.2 Get batch pad word length
                pad_context_word_len = min(self.max_word_len, max([max(sample['context_word_len']) for sample in batch]))
                pad_question_word_len = min(self.max_word_len, max([max(sample['question_word_len']) for sample in batch]))

                # 2.3 Padding batch word len to pad word length
                for sample in batch:
                    sample['context_char_ids'] = [BatchGenerator._dynamic_padding(char_ids, pad_context_word_len, char_pad_idx)
                                                  for char_ids in sample['context_char_ids']]
                    sample['question_char_ids'] = [BatchGenerator._dynamic_padding(char_ids, pad_question_word_len, char_pad_idx)
                                                   for char_ids in sample['question_char_ids']]
                    sample['context_word_len'] = [min(word_len, pad_context_word_len)
                                                  for word_len in sample['context_word_len']]
                    sample['question_word_len'] = [min(word_len, pad_question_word_len)
                                                   for word_len in sample['question_word_len']]

            # 3. Convert batch data to `torch tensor`
            for sample in batch:
                for key, value in sample.items():
                    result[key].append(value)
            for key, value in result.items():
                result[key] = torch.tensor(value)

            return result

        return DataLoader(dataset=self.dataset, shuffle=self.training,
                          batch_size=self.batch_size,
                          collate_fn=mrc_collate,
                          num_workers=self.num_parallel_calls)


class MRCDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)
