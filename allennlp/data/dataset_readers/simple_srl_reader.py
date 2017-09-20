# Simple reader for SRL data. Modified from allennlp's seq2seq reader
# and SRL reader.

from typing import Dict, List
import logging

from overrides import overrides
import tqdm

import codecs
import os

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_DEFAULT_WORD_TAG_DELIMITER = '|||'


@DatasetReader.register("simple_srl_reader")
class SimpleSrlReader(DatasetReader):
  def __init__(self,
               word_tag_delimiter: str = _DEFAULT_WORD_TAG_DELIMITER,
               token_indexers: Dict[str, TokenIndexer] = None) -> None:
    self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    self._word_tag_delimiter = word_tag_delimiter

  @overrides
  def read(self, file_path):
    # if `file_path` is a URL, redirect to the cache
    file_path = cached_path(file_path)

    with open(file_path, "r") as data_file:
      instances = []
      logger.info("Reading instances from lines in file at: %s", file_path)
      for line in tqdm.tqdm(data_file):
        line = line.strip("\n")
        # skip blank lines
        if not line:
          continue

        pred_id = int(line.split()[0])
        tokens_and_tags = line.split(maxsplit=1)[1].split(self._word_tag_delimiter)
        tokens = [Token(token) for token in tokens_and_tags[0].split()]
        tags = [tag for tag in tokens_and_tags[1].split()]

        pred_tags = [0 if i != pred_id else 1 for i in range(len(tokens))]
        sequence = TextField(tokens, self._token_indexers)
        sequence_tags = SequenceLabelField(tags, sequence)
        sequence_pred_tags = SequenceLabelField(pred_tags, sequence)

        instances.append(Instance({'tokens': sequence,
                                   'tags': sequence_tags,
                                   'verb_indicator': sequence_pred_tags }))
        if not instances:
          raise ConfigurationError("No instances were read from the given filepath {}. "
                                   "Is the path correct?".format(file_path))
    return Dataset(instances)

  def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
    """
    We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
    """
    # pylint: disable=arguments-differ
    return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

  @classmethod
  def from_params(cls, params: Params) -> 'SimpleSrlReader':
    token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
    word_tag_delimiter = params.pop("word_tag_delimiter", _DEFAULT_WORD_TAG_DELIMITER)
    params.assert_empty(cls.__name__)
    return SimpleSrlReader(token_indexers=token_indexers,
                           word_tag_delimiter=word_tag_delimiter)

