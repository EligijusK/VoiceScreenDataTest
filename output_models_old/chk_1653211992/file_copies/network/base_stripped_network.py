import abc
import torch
import numpy as np


class BaseStrippedNetwork(abc.ABC):
    def __init__(self, *, path_weights: str, device: str = "cuda", **kwargs):
        super(BaseStrippedNetwork, self).__init__()
        
        model = self._init_model(**kwargs)

        self.model = model.cuda() if device == "cuda" else model.cpu()
        self.device = device

        self.restore(path_weights)

    @abc.abstractmethod
    def _init_model(*args, **kwargs):
        pass

    def restore(self, path_weights):
        if path_weights.endswith(".xth"):
            import pickle
            from cryptography.fernet import Fernet
            from . import encryption_key

            with open(path_weights, "rb") as f:
                encrypted_bytes = f.read()

                fernet = Fernet(encryption_key)
                decrypted = fernet.decrypt(encrypted_bytes)

                weights = pickle.loads(decrypted)

                for key in weights:
                    t = torch.tensor(weights[key], device=self.device)
                    weights[key] = t

                self.model.load_state_dict(weights)
        else:
            state = torch.load(path_weights, map_location=self.device)
            self.model.load_state_dict(state["state_model"])

    def predict(self, t_in: np.array) -> np.array:
        model = self.model

        model.eval()

        with torch.no_grad():
            t_in = t_in.float()
            t_in = t_in.cuda().contiguous() if self.device == "cuda" else t_in

            return model(t_in)