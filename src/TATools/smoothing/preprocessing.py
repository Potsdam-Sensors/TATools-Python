from .normalization import Normalization, Data
from .smoothing import Smoothing, SmoothingOp
from typing import Union, List, Tuple

class DataPreprocessor(object):
    def __init__(self, *operations):
        for op in operations:
            if op is None: continue
            assert isinstance(op, (Smoothing, SmoothingOp, Normalization))
        self.operations: List[Union[Smoothing, SmoothingOp, Normalization]] = [op for op in operations if op is not None]
    
    def normalizes(self) -> bool:
        return any([isinstance(op, Normalization) for op in self.operations])
    
    def smooths(self) -> bool:
        return any([isinstance(op, (Smoothing, SmoothingOp)) for op in self.operations])
    
    def apply(self, data: Data) -> Data:
        for op in self.operations:
            if isinstance(op, (Smoothing, SmoothingOp)):
                data = op.smooth(data)
            elif isinstance(op, (Normalization)):
                data = op.apply(data)
            else:
                raise ValueError()
        return data
    
    def label(self, delim: str = "\n") -> str:
        if len(self.operations) == 0:
            return ""
        ret = self.operations[0].label()
        for op in self.operations[1:]:
            ret += delim
            ret += op.label()
        return ret
    
    def filename_label(self, delim: str = "_") -> str:
        if len(self.operations) == 0:
            return ""
        ret = self.operations[0].filename_label()
        for op in self.operations[1:]:
            ret += delim
            ret += op.filename_label()
        return ret