from abc import ABC, abstractmethod


class AbstractTarget(ABC):

    @abstractmethod
    def get_contexts(self, n_context):
        pass

    @abstractmethod
    def log_prob_tgt(self, contexts, samples):
        pass
