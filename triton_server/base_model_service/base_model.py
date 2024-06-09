from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def set_up(self) -> None:
        """Load best  model"""
        raise NotImplementedError

    @abstractmethod
    def infer(self, input_data) -> dict:
        """Infer the model"""
        raise NotImplementedError
