import logging
from typing import Dict, Optional

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.initializers import InitializerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("sent_rep")
class SentenceRepresentationModel(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.
    sentence_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 aggregate_feedforward: FeedForward,
                 initializer: InitializerApplicator,
                 srl_model: SemanticRoleLabeler,
                 sentence_encoder: Optional[Seq2SeqEncoder] = None) -> None:
        super(SentenceRepresentationModel, self).__init__(vocab)

        self._srl_model = srl_model
        self._text_field_embedder = text_field_embedder
        self._aggregate_feedforward = aggregate_feedforward
        self._sentence_encoder = sentence_encoder
        self._num_labels = vocab.get_vocab_size(namespace="labels")

        # TODO: Check dimension with SRL embeddings.
        #if text_field_embedder.get_output_dim() != attend_feedforward.get_input_dim():
        #    raise ConfigurationError("Output dimension of the text_field_embedder (dim: {}), "
        #                             "must match the input_dim of the FeedForward layer "
        #                             "attend_feedforward, (dim: {}). ".format(text_field_embedder.get_output_dim(),
        #                                                                      attend_feedforward.get_input_dim()))
        #if aggregate_feedforward.get_output_dim() != self._num_labels:
        #    raise ConfigurationError("Final output dimension (%d) must equal num labels (%d)" %
        #                             (aggregate_feedforward.get_output_dim(), self._num_labels))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                # premise_verbs: torch.LongTensor,
                # hypothesis_verbs: torch.LongTensor,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        premise_verbs : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis_verbs : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._sentence_encoder:
            embedded_premise = self._sentence_encoder(embedded_premise, premise_mask)
            embedded_hypothesis = self._sentence_encoder(embedded_hypothesis, hypothesis_mask)

        # Compute SRL encoding. Shape: (batch_size, premise|hypothesis-length, SRL encoding size)
        # srl_premise = self._srl_model.forward(premise, premise_verbs)["encoded_text"]
        # srl_hypothesis = self._srl_model.forward(hypothesis, hypothesis_verbs)["encoded_text"]

        # Do not backpropagate through the SRL representation.
        # srl_embedded_premise = torch.cat([embedded_premise, srl_premise.detach()], dim=-1)
        # srl_embedded_hypothesis = torch.cat([embedded_hypothesis, srl_hypothesis.detach()], dim=-1)

        # masked_premise = embedded_premise - float('inf') * (1 - premise_mask.unsqueeze(-1))
        # masked_hypothesis = embedded_hypothesis - float('inf') * (1 - hypothesis_mask.unsqueeze(-1))
        masked_premise = embedded_premise * premise_mask.unsqueeze(-1)
        masked_hypothesis = embedded_hypothesis * hypothesis_mask.unsqueeze(-1)

        # Max pooling along the time dimension.
        # compared_premise, _ = masked_premise.max(dim=1)
        # compared_hypothesis, _ = masked_hypothesis.max(dim=1)
        compared_premise = masked_premise.sum(dim=1)
        compared_hypothesis = masked_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([
            compared_premise, compared_hypothesis,
            torch.abs(compared_premise - compared_hypothesis),
            compared_premise * compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SentenceRepresentationModel':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        sentence_encoder_params = params.pop("sentence_encoder", None)
        if sentence_encoder_params is not None:
            sentence_encoder = Seq2SeqEncoder.from_params(sentence_encoder_params)
        else:
            sentence_encoder = None

        #hypothesis_encoder_params = params.pop("hypothesis_encoder", None)
        #if hypothesis_encoder_params is not None:
        #    hypothesis_encoder = Seq2SeqEncoder.from_params(hypothesis_encoder_params)
        #else:
        #    hypothesis_encoder = None

        #srl_model_archive = params.pop('srl_model_archive', None)
        #if srl_model_archive is not None:
        #    logger.info("Loaded pretrained SRL model from {}".format(srl_model_archive))
        #    archive = load_archive(srl_model_archive)
        #    srl_model = archive.model
        #else:
        srl_model = None

        aggregate_feedforward = FeedForward.from_params(params.pop('aggregate_feedforward'))
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   aggregate_feedforward=aggregate_feedforward,
                   initializer=initializer,
                   srl_model=srl_model,
                   sentence_encoder=sentence_encoder)
