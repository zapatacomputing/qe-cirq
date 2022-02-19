# qe-cirq

[![codecov](https://codecov.io/gh/zapatacomputing/qe-cirq/branch/main/graph/badge.svg?token=9513V4OWNI)](https://codecov.io/gh/zapatacomputing/qe-cirq)

An Orquestra Quantum Engine Resource for Cirq

## Overview

`qe-cirq` is a Python module that exposes Cirq's simulators as a [`z-quantum-core`](https://github.com/zapatacomputing/z-quantum-core/blob/master/src/python/zquantum/core/interfaces/backend.py) `QuantumSimulator`. The simulator can be easily imported using:

```
from qecirq.simulator import CirqSimulator
```

In addition, it interfaces with the noise models and provides converters that allow switching between `cirq` circuits and those of `z-quantum-core`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Cirq integration docs](http://docs.orquestra.io/other-resources/framework-integrations/cirq/).
For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).
