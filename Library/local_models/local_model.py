from abc import ABC, abstractmethod
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

class LocalModel(ABC):

    model: Optional[PreTrainedModel]
    processor: Optional[ProcessorMixin]
    tokenizer: Optional[PreTrainedTokenizerBase]
    base_model: str
    fine_tuned_path: Optional[str]
    eos_token: str
    pad_token: str
    prepared_for_inference: bool = False

    def __init__(self, base_model, fine_tuned_path=""):
        self.base_model = base_model
        self.fine_tuned_path = fine_tuned_path if fine_tuned_path else None
        self.model = None
        self.init_model()
        self.init_processor()
        self.init_tokenizer()
        self.init_special_tokens()

    @abstractmethod
    def init_model(self): ...

    @abstractmethod
    def init_processor(self): ...

    @abstractmethod
    def init_tokenizer(self): ...

    @abstractmethod
    def init_special_tokens(self): ...

    @abstractmethod
    def run_inference(self, text: str, images: list, messages: list): ...


