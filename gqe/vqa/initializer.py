from qml.core.initializer import VQAInitializer
from qswift.initializer import CircuitInitializer
from qwrapper.circuit import init_circuit


class InitializerDelegate(VQAInitializer):
    def __init__(self, initializer: CircuitInitializer, nqubit, tool="qulacs"):
        self.initializer = initializer
        self.nqubit = nqubit
        self.tool = tool

    def initialize(self):
        return self.initializer.init_circuit(self.nqubit, [], self.tool)
