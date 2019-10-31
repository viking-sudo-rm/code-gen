from typing import Dict
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)


@DatasetReader.register("char_lm")
class CharLanguageModelingReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(file_path) as fh:
            for line in fh.readlines():
                line = line.strip()
                if len(line) < 2:
                    continue
                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        tokens = [Token("<s>")]
        tokens.extend(Token(c) for c in sentence)
        tokens.append(Token("</s>"))

        return Instance({
            "source": TextField(tokens, token_indexers=self._token_indexers)
        })
