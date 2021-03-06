"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of :class:`~allennlp.models.model.Model`.
"""

from allennlp.models.archival import archive_model, load_archive
from allennlp.models.decomposable_attention import DecomposableAttention
from allennlp.models.decacc_srl import DecAccSRL
from allennlp.models.sent_rep import SentenceRepresentationModel
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.models.simple_tagger import SimpleTagger
