import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import numpy as np
import collections
import logging

import multiprocessing
class BatchGenerator(object):
    def __init__(self, vocab, instances, batch_size=32, use_char=True, training=False,
                 additional_fields=None, feature_vocab=None,num_parallel_calls=0,shuffle_ratio=1.0):
        self.instances = instances
        self.vocab = vocab
        self.batch_size = batch_size
        self.use_char = use_char
        self.training = training
        self.shuffle_ratio = shuffle_ratio
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls>0 else multiprocessing.cpu_count()//2
        self.additional_fields = additional_fields if additional_fields is not None else list()
        self.feature_vocab = feature_vocab if feature_vocab is not None else dict()

        if self.instances is None or len(self.instances) == 0:
            raise ValueError('empty instances!!')

        self.dataset = self.build_input_pipeline()
        print(self.dataset[0].keys())
        print(self.dataset[0])

        def mrc_collate(batch, word_pad_idx=self.vocab.get_word_pad_idx()):
            result = {}
            for key in batch[0].keys():
                result[key] = []
            # get batch pad length, TODO 还没做截断
            pad_context_len = max([sample['context_len'] for sample in batch])
            pad_question_len = max([sample['question_len'] for sample in batch])
            # padding context and question
            for sample in batch:
                sample['context_ids'] = (sample['context_ids'] + [word_pad_idx] *
                                            (pad_context_len - sample['context_len']))[:pad_context_len]
                sample['question_ids'] = (sample['question_ids'] + [word_pad_idx] *
                                            (pad_question_len - sample['question_len']))[:pad_question_len]
            for sample in batch:
                for key, value in sample.items():
                    result[key].append(value)
            for key, value in result.items():
                result[key] = torch.tensor(value)
            return result

        self.dataloader = DataLoader(dataset=self.dataset,shuffle=training,
                                      batch_size=self.batch_size,
                                      collate_fn=mrc_collate,
                                      num_workers=self.num_parallel_calls)

    def _generator(self, dataloader):
        for batch_data in dataloader:
            yield batch_data

    def init(self):
        self.generator = self._generator(self.dataloader)

    def next(self):
        if self.generator is None:
            raise Exception('you must do init before do next.')
        return next(self.generator)

    def get_dataset_size(self):
        return len(self.dataset)

    def get_batch_size(self):
        return self.batch_size

    def get_raw_dataset(self):
        return self.instances

    # def get_dataset(self):
    #     return self.dataset
    #
    # def get_dataloader(self):
    #     return self.dataloader

    @staticmethod
    def detect_input_type(instance, additional_fields=None):
        instance_keys = instance.keys()
        fields = ['context_tokens', 'question_tokens', 'answer_start', 'answer_end']
        try:
            for f in fields:
                assert f in instance_keys
        except:
            raise ValueError('A instance should contain at least "context_tokens", "question_tokens", \
                             "answer_start", "answer_end" four fields!')

        if additional_fields is not None and isinstance(additional_fields, list):
            fields.extend(additional_fields)

        def get_type(value):
            if isinstance(value, float):
                return torch.float32
            elif isinstance(value, int):
                return torch.int32
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

    def build_input_pipeline(self):
        input_fields, input_type_dict = BatchGenerator.detect_input_type(self.instances[0], self.additional_fields)

        # 1. Get data
        # def make_generator():
        #     for instance in self.instances:
        #         new_dict = {k: instance[k] for k in input_fields}
        #         yield new_dict
        #
        # dataset = tf.data.Dataset.from_generator(make_generator,
        #                                          {w: input_type_dict[w] for w in input_fields},
        #                                          {w: input_shape_dict[w] for w in input_fields}
        #                                          )

        filtered_instances = [{k: instance[k] for k in input_fields} for instance in self.instances]

        # 2. Character extracting function
        # def extract_char(token, default_value="<PAD>"):
        #     out = tf.string_split(token, delimiter='')
        #     out = tf.sparse.to_dense(out, default_value=default_value)
        #     return out

        # 3. Build look-up table from vocabulary
        # 3.1 Word look-up table
        # word_vocab = self.vocab.get_word_vocab()
        # word_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(word_vocab), num_oov_buckets=1)
        # 3.2 Char look-up table
        # if self.use_char:
            # char_vocab = self.vocab.get_char_vocab()
            # char_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(char_vocab), num_oov_buckets=1)
        # 3.3 other feature look-up table
        # if len(self.feature_vocab) > 0:
        #     feature_table = {}
        #     for feature_name, vocab in self.feature_vocab.items():
        #         feature_table[feature_name] = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab),
        #                                                                                 num_oov_buckets=1)

        # 4. Some preprocessing, including char extraction, lowercasing, length
        def transform_new_instance(instance, input_type_dict):
            context_tokens = instance['context_tokens']
            question_tokens = instance['question_tokens']

            # if self.use_char:
            #     context_char = extract_char(context_tokens)
            #     context_word_len = tf.strings.length(context_tokens)
            #     question_char = extract_char(question_tokens)
            #     instance['context_char'] = tf.cast(char_table.lookup(context_char), tf.int32)
            #     instance['question_char'] = tf.cast(char_table.lookup(question_char), tf.int32)

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
        new_instances = [transform_new_instance(instance, input_type_dict) for instance in filtered_instances]

        # 6. Padding and batching
        # def build_padded_shape(output_shapes):
        #     padded_shapes = dict()
        #     for field, shape in output_shapes.items():
        #         field_dim = len(shape.as_list())
        #         if field_dim > 0:
        #             padded_shapes[field] = tf.TensorShape([None] * field_dim)
        #         else:
        #             padded_shapes[field] = tf.TensorShape([])
        #     return padded_shapes
        #
        # padded_shapes = build_padded_shape(dataset.output_shapes)
        # dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)


        return MRCDataset(new_instances)


class MRCDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)
