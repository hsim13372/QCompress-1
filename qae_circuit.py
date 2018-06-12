
"""Class for generating a unitary circuit for quantum autoencoders."""

from pyquil.gates import RX, RY, RZ
from pyquil.quil import Program
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_sin, quil_cos

import numpy as np


class QAECircuit(object):
    """Class to build a quantum autoencode.

    Generates a program that performs a parameterized rotation gate on all the input qubits,
    followed by all possible combinations of parameterized controlled rotation gates and then
    another single qubit rotation gate on all qubits. It then daggers and "flips" (reverses qubits)
    and appends it to the original program.

    Attributes
    ----------
    theta: pyquil.parameters.Parameter
        Parameter for defining parameterized gates
    crx: numpy.Array
        Matrix representation of controlled RX gate
    crx_def: pyquil.quilbase.DefGate
        Controlled RX gate definition
    CRX: pyquil.quilbase.Gate
        pyquil Gate for parameterized RX
    cry: numpy.Array
        Matrix representation of controlled RY gate
    cry_def: pyquil.quilbase.DefGate
        Controlled RY gate definition
    CRY: pyquil.quilbase.Gate
        pyquil Gate for parameterized RY
    crz: numpy.Array
        Matrix representation of controlled RZ gate
    crz_def: pyquil.quilbase.DefGate
        Controlled RZ gate definition
    CRZ: pyquil.quilbase.Gate
        pyquil Gate for parameterized RZ
    gates_dict : dictionary
        Dictionary of string representation => pyquil.quilbase.Gate. Contains all rotation
        gates and controlled rotation gates

    """

    theta = Parameter('theta')
    crx = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
        [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]
    ])
    crx_def = DefGate('CRX', crx, [theta])
    CRX = crx_def.get_constructor()

    cry = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, quil_cos(theta / 2), -1 * quil_sin(theta / 2)],
        [0, 0, quil_sin(theta / 2), quil_cos(theta / 2)]
    ])
    cry_def = DefGate('CRY', cry, [theta])
    CRY = cry_def.get_constructor()

    crz = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, quil_cos(theta / 2) - 1j * quil_sin(theta / 2), 0],
        [0, 0, 0, quil_cos(theta / 2) + 1j * quil_sin(theta / 2)]
    ])
    crz_def = DefGate('CRZ', crz, [theta])
    CRZ = crx_def.get_constructor()

    gates_dict = {
        'RX': RX, 'CRX': CRX,
        'RY': RY, 'CRY': CRY,
        'RZ': RZ, 'CRZ': CRZ,
    }

    def __init__(self, num_qubits, num_latent_qubits, thetas, axes=None, qubits=None):
        """Initialize the circuit.

        Parameters
        ----------
        num_qubits : int, required
            Total number of qubits required by the circuit.
            This should be (2 * num_input_qubits - num_latent_qubits)
        num_latent_qubits : int, required
            Number of latent space qubits
        thetas : list[numeric], required
            List of rotation parameters for each gate this should be of length 2n + n(n-1)
            where n is the number of input qubits
        axes : list[string], optional (default None)
            List of rotation directions per block of gates. Possible values are 'X', 'Y', 'Z'
            Sets all values to 'X' when not set
        qubits : list[int], optional (default None)
            List of qubit labels for the circuit
            Set to 0 - (num_qubits - 1) when not passed
            Useful to optimize the circuit based on topology of a QPU

        Attributes
        ----------
        num_input_qubits: int
            Number of qubits input to the first portion of the circuit.

        """
        self.num_qubits = num_qubits
        self.num_latent_qubits = num_latent_qubits
        self.num_input_qubits = int((num_qubits + num_latent_qubits) / 2)

        if qubits is None:
            self.qubits = []
            for i in range(self.num_qubits):
                self.qubits.append(i)
        else:
            self.qubits = qubits

        if axes is None:
            self.axes = ['X'] * (self.num_input_qubits + 2)
        else:
            self.axes = axes

        self.thetas = thetas

        self.program = Program()

    def def_gates_program(self):
        """Create a program with defined with controlled rotation gates.

        Returns
        -------
        pyquil.quil.Program
            pyquil Program with defined controlled rotation gates

        """
        return Program(QAECircuit.crx_def, QAECircuit.cry_def, QAECircuit.crz_def)

    def build_circuit(self):
        """Build quantum autoencoder circuit.

        Returns
        -------
        pyquil.quil.Program
            pyquil Program

        """
        self.program += self.def_gates_program()

        qubits = self.qubits[:self.num_input_qubits]
        thetas = self.thetas[:self.num_input_qubits]

        axis = self.axes[0]
        self.program += self.build_rotation_block(axis, qubits, thetas)
        for i in range(self.num_input_qubits):
            axis = self.axes[i+1]
            control_qubit = qubits[i]
            start_theta = self.num_input_qubits + (self.num_input_qubits - 1) * i
            end_theta = start_theta + (self.num_input_qubits - 1)

            thetas = self.thetas[start_theta:end_theta]

            self.program += self.build_controlled_rotation_block(axis, qubits, thetas, control_qubit)

        axis = self.axes[self.num_input_qubits+1]
        start_theta = self.num_input_qubits + (self.num_input_qubits - 1) * self.num_input_qubits
        end_theta = start_theta + self.num_input_qubits
        thetas = self.thetas[start_theta:end_theta]

        self.program += self.build_rotation_block(axis, qubits, thetas)
        return self.program + self.dagger_and_flip(self.program, self.qubits)

    def build_rotation_block(self, axis, qubits, thetas):
        """Build a circuit block made from single qubit rotation gates.

        Parameters
        ----------
        axis : string, required
            Axis for the rotation gates. 'X', 'Y', or 'Z'
        qubits : list[int], required
            List of qubit labels to attach the gates to
        thetas : list[numeric], required
            List of rotation paramaters for each gate. Length should match that of qubits

        Returns
        -------
        pyquil.quil.Program
            pyquil Program that consists of rotation gates acting on passed in qubits

        """
        gate = self.gate_from_axis(axis, 1)
        program = Program()
        for i, qubit in enumerate(qubits):
            program += Program(gate(thetas[i], qubit))

        return program

    def build_controlled_rotation_block(self, axis, qubits, thetas, qubit_control):
        """Build a circuit block made from two qubit controlled rotation gates.

        Parameters
        ----------
        axis : string, required
            Axis for the rotation gates. 'X', 'Y', or 'Z'
        qubits : list[int], required
            List of qubit labels to attach the gates to
        thetas : list[numeric], required
            List of rotation paramaters for each gate. Length should match that of qubits
        qubit_control : int, required
            Qubit that acts as the control on the controlled rotation gates

        Returns
        -------
        pyquil.quil.Program
            pyquil Program that consists of controlled rotation gates acting on passed in qubits

        """
        gate = self.gate_from_axis(axis, 2)
        program = Program()

        i = 0
        for qubit in qubits:
            if qubit != qubit_control:
                theta = thetas[i]
                i += 1
                program += Program(gate(theta)(qubit_control,  qubit))

        return program

    def gate_from_axis(self, axes, num_qubits):
        """Get a gate based on an axis and number of qubits it acts on.

        Parameters
        =====
        axes : string, required
            Axis rotation gate to fetch
        num_qubits : int, required
            Number of qubits the gate acts on (1 or 2)

        Returns
        -------
        pyquil.quilbase.Gate
            Rotation gate or controlled rotation gate based on axis and number of qubits

        """
        gate_str = "R{}".format(axes)
        if 2 == num_qubits:
            gate_str = "C" + gate_str
        return QAECircuit.gates_dict[gate_str]

    def dagger_and_flip(self, program, qubits):
        """Build and return a daggered and flipped version of a program.

        Parameters
        ----------
        program : pyquil.quil.Program
            Program to be daggered and flipped
        qubits : type
            List of qubits to be used when flipping the programself.
            e.g. [1, 2, 3, 4] are used in the program, they will be replaced with [4, 3, 2, 1] in
            the flipped version. If a list larger than the number of qubits is passed, the mapping
            accounts for this. e.g. qubits [1, 2, 3, 4] are using by `program` but `qubits` passed
            is [1, 2, 3, 4, 5, 6] in the returned program, [6, 5, 4, 3] will only be used. The
            mapping is as follows: 1 => 6, 2 => 5, 3 => 4, 4 => 2.

        Returns
        -------
        pyquil.quil.Program
            A flipped and daggered program

        Notes
        -----
        This only works for rotation and controlled rotation gates. See pyquil.quil.Program.dagger()
        for further implmementation

        """
        qubits_reversed = dict(zip(qubits, reversed(qubits)))
        flipped_program = Program()
        for gate in reversed(program.instructions):
            gate_qubits = []
            for i in gate.qubits:
                gate_qubits.append(qubits_reversed[i.index])

            # Daggered versions of all gates are just a rotation in the opposite direction
            negated_params = list(map(lambda x: -1 * x, gate.params))
            flipped_program.inst(QAECircuit.gates_dict[gate.name](*negated_params)(*gate_qubits))

        return flipped_program


if __name__ == '__main__':
    axes = ['X', 'Y', 'X', 'Y', 'X', 'Y']
    thetas = [
        np.pi, np.pi, np.pi, np.pi,
        np.pi/2, np.pi/2, np.pi/2,
        np.pi/2, np.pi/2, np.pi/2,
        np.pi/2, np.pi/2, np.pi/2,
        np.pi/2, np.pi/2, np.pi/2,
        np.pi, np.pi, np.pi, np.pi,
    ]

    qae = QAECircuit(num_qubits=7, num_latent_qubits=1, thetas=thetas, axes=axes)
    p = qae.build_circuit()
    print(p)
