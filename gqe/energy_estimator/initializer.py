from qwrapper.circuit import QWrapper


class Initializer:
    def initialize(self, qc, targets) -> QWrapper:
        return qc


class XInitializer(Initializer):
    def initialize(self, qc, targets) -> QWrapper:
        for t in targets:
            qc.h(t)
        return qc