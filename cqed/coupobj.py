# This file is part of cqed: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
########################################################################
"""The CoupledObjects class for representing a system of interacting
superconducting qubits and resonators.
"""

__all__ = ['CoupledObjects']

import numpy as np
import itertools

import qutip as qt


class CoupledObjects(object):
    """A class for representing interacting qubits and resonators.

    Facilitates work in the tensor-product Hilbert space and allows
    adressing interacting eigenstates by their noninteracting labels
    such as in system.freq('00', '10') for the frequency of the 00-10
    transition in a two-qubit system. Unless specified, all returned
    quantum states and operators are in the noninteracting eigenbasis,
    which is generated from the tensor product of individual eigenbases.

    Parameters
    ----------
    *args
        Individual objects of qubit and cavity types and interactions
        between them. A coupling between two objects `obj1` and `obj2`
        is represented by a list `[obj1, obj2, E_int, coupling_type]`,
        where `E_int` and `coupling_type` are the interaction strength
        and type. Acceptable values of coupling_type:
        - 'charge' for the capacitive coupling term E_int * n_A * n_B,
        - 'flux' for the inductive coupling term E_int * phi_A * phi_B,
        - 'JC-charge' for E_int * n * (a+a^+) type of coupling,
        - 'JC-rwa' for the Jaynes-Cummings term in the RWA
              as in E_int * (a*b^+ + a^+*b),
        - 'JC-full' for the full Jaynes-Cummings term
              as in -E_int * (a-a^+) * (b-b^+).
        (Here n, phi, and a are the charge, flux, and annihilation
         operators of a qubit or cavity; the qubit or cavity objects
         must have the corresponding methods.)

    Examples
    --------
    1) A system of two inductively coupled fluxonium qubits.

    >>> qubit1 = fluxonium.Fluxonium(E_L1, E_C1, E_J1)
    >>> qubit2 = fluxonium.Fluxonium(E_L2, E_C2, E_J2)
    >>> system = coupobj.CoupledObjects(
    ...     qubit1, qubit2, [qubit1, qubit2, E_int, 'flux'])

    2) A system of two qubits with both capacitive and inductive
    couplings.

    >>> system = coupobj.CoupledObjects(
    ...     qubit1, qubit2,
    ...     [qubit1, qubit2, E_int_C, 'charge'],
    ...     [qubit1, qubit2, E_int_L, 'flux'])

    3) A microwave resonator coupled to a qubit via
    g*(a + a^+)*n interaction.

    >>> resonator = cavity.Cavity(omega=omega_c, nlev=nlev_cav)
    >>> qubit = fluxonium.Fluxonium(E_L, E_C, E_J)
    >>> system = coupobj.CoupledObjects(
    ...     resonator, qubit,[resonator, qubit, g, 'JC-charge'])
    """

    def __init__(self, *args):
        self._objects = []  # Qubit and resonator objects.
        self._couplings = []  # Coupling terms.
        for arg in args:
            if isinstance(arg, list):
                self._couplings.append(arg)
            else:
                self._check_obj(arg)
                self._objects.append(arg)
        self._nobj = len(self._objects)
        self._ncoupl = len(self._couplings)
        self._reset_cache()

    def _check_obj(self, obj):
        if (not hasattr(obj, 'type') or obj.type != 'qubit'
                and obj.type != 'cavity'):
            raise Exception('The object parameter is unrecognized.')

    def _reset_cache(self):
        """Resets cached data."""
        self._eigvals = None  # Interacting eigenvalues.
        self._eigvecs = None  # Interacting eigenvectors.
        self._state_labels = None  # Labels of interacting states.
        self._eigvals_nonint = None  # Noninteracting eigenvalues.
        self._eigvecs_nonint = None  # Noninteracting eigenvectors.
        self._state_labels_nonint = None  # Labels of noninteracting states.
        self._nlev = np.prod([obj.nlev for obj in self._objects])

    def reset(self):
        """Manual "reset" of the cached data in a class instance.

        It is necessary to call this method after a change in any
        parameter that is directly or indirectly used in the class
        instance. For example, after any of the attributes of
        an underlying qubit object has been changed, certain cached
        data such as eigenvalues have to be re-calculated.

        May be deprecated in the future.
        """
        self._reset_cache()

    def promote_op(self, obj, operator):
        """Rewrites an operator in the tensor-product Hilbert space.

        Parameters
        ----------
        obj : object or int
            Qubit or cavity object or the sequential number of the
            object in the class instance initialization.
        operator : :class:`qutip.Qobj`
            The operator written in the Hilbert space of `obj`.

        Returns
        -------
        :class:`qutip.Qobj`
            The operator `operator` rewritten in the tensor-product
            Hilbert space of the composite system.

        See Also
        --------
        phi, n, a, adag : shortcuts for the most popular operators.

        Examples
        --------
        (Just for illustration, self.n() method is more convenient.)

        >>> system = coupobj.CoupledObjects(
        ...     qubit1, qubit2, [qubit1, qubit2, J_C, 'charge'])
        >>> n1 = system.promote_op(qubit1, qubit1.n())
        >>> n2 = system.promote_op(2, qubit2.n())
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        if obj not in self._objects:
            raise Exception('The object parameter is unrecognized.')
        if operator.dims != obj.H().dims:
            raise Exception(
                    'The operator does not agree with its underlying object.')
        obj_index = self._objects.index(obj)
        # Add identity operators from the left and from the right.
        for k in range(obj_index-1, -1, -1):
            operator = qt.tensor(qt.qeye(self._objects[k].nlev), operator)
        for k in range(obj_index+1, self._nobj):
            operator = qt.tensor(operator, qt.qeye(self._objects[k].nlev))
        return operator

    def phi(self, obj):
        """The flux operator for a qubit in the tensor-product space.

        See Also
        --------
        promote_op

        Examples
        --------
        >>> system = coupobj.CoupledObjects(
        ...     qubit1, qubit2, [qubit1, qubit2, J_C, 'charge'])
        >>> phi1 = system.phi(qubit1)
        >>> phi2 = system.phi(2)
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.phi())

    def n(self, obj):
        """The charge operator for a qubit in the tensor-product space.

        See Also
        --------
        promote_op, phi
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.n())

    def a(self, obj):
        """The annihilation operator in the tensor-product space.

        See Also
        --------
        promote_op, phi
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.a())

    def adag(self, obj):
        """The creation operator in the tensor-product space.

        See Also
        --------
        promote_op, phi
        """
        if isinstance(obj, int):
            if obj >= self._nobj or obj < 0:
                raise Exception('The object number is out of bounds.')
            obj = self._objects[obj]
        return self.promote_op(obj, obj.a().dag())

    def H_0(self):
        """The Hamiltonian of uncoupled quantum objects."""
        H_0 = 0
        for obj in self._objects:
            H_0 += self.promote_op(obj, obj.H())
        return H_0

    def V(self):
        """The coupling part of the Hamiltonian."""
        V = 0
        for coupling_term in self._couplings:
            obj1 = coupling_term[0]
            obj2 = coupling_term[1]
            E_int = coupling_term[2]
            if isinstance(coupling_term[3], str):
                if coupling_term[3] == 'flux':
                    # Qubit-qubit flux coupling.
                    op1 = self.promote_op(obj1, obj1.phi())
                    op2 = self.promote_op(obj2, obj2.phi())
                elif coupling_term[3] == 'charge':
                    # Qubit-qubit charge coupling.
                    op1 = self.promote_op(obj1, obj1.n())
                    op2 = self.promote_op(obj2, obj2.n())
                elif coupling_term[3] == 'JC-charge':
                    # Qubit-resonator coupling via charge.
                    if obj1.type == 'cavity':
                        op1 = obj1.a() + obj1.adag()
                        op2 = obj2.n()
                    else:
                        op1 = obj1.n()
                        op2 = obj2.a() + obj2.adag()
                    op1 = self.promote_op(obj1, op1)
                    op2 = self.promote_op(obj2, op2)
                elif coupling_term[3] == 'JC-rwa':
                    # Jaynes-Cummings coupling in the RWA.
                    op1 = self.promote_op(obj1, obj1.a())
                    op2 = self.promote_op(obj2, obj2.a())
                elif coupling_term[3] == 'JC-full':
                    # Jaynes-Cummings term, no RWA.
                    op1 = obj1.a() - obj1.a().dag()
                    op2 = -(obj2.a() - obj2.a().dag())
                    op1 = self.promote_op(obj1, op1)
                    op2 = self.promote_op(obj2, op2)
                else:
                    raise Exception(
                            'This type of coupling is not implemented yet.')
                if coupling_term[3] == 'JC-rwa':
                    V += E_int * (op1 * op2.dag() + op1.dag() * op2)
                else:
                    V += E_int * op1 * op2
            else:
                raise Exception(
                        'This type of coupling is not implemented yet.')
        return V

    def H(self):
        """The Hamiltonian of the interacting system."""
        return self.H_0() + self.V()

    def _spectrum_nonint(self):
        """Spectrum and level labels in the absence of coupling.

        Returns
        -------
        1d array
            Noninteracting energies in ascending order.
        1d array of :class:`qutip.Qobj`
            Corresponding noninteracting eigenvectors.
        2d array
            State labels in the same order as energies.
            1st column - tuple-like labels such as (0, 1)
                for the state '01' of a two-qubit system.
            2nd column - string labels such as '01'.
        """
        objects = self._objects
        if (self._eigvals_nonint is None or self._eigvecs_nonint is None
                or self._state_labels_nonint is None):
            nobj = self._nobj
            states = np.empty((self._nlev, 4), dtype=object)
            # To make the loop over all possible combinations of indices.
            iterable = [range(obj.nlev) for obj in objects]
            ind = 0
            for state_tuple in itertools.product(*iterable):
                # Noninteracting energy is the sum of energies.
                states[ind, 0] = np.sum(
                        objects[k].level(state_tuple[k]) for k in range(nobj))
                eigenvector = qt.basis(objects[0].nlev, state_tuple[0])
                for k in range(1, nobj):
                    eigenvector = qt.tensor(eigenvector, qt.basis(
                            objects[k].nlev, state_tuple[k]))
                states[ind, 1] = eigenvector
                states[ind, 2] = state_tuple
                # The string label such as '01' for a two-qubit system.
                states[ind, 3] = ''.join(str(k) for k in state_tuple)
                ind += 1
            states = states[np.argsort(states[:, 0])]
            self._eigvals_nonint = states[:, 0]
            self._eigvecs_nonint = states[:, 1]
            self._state_labels_nonint = states[:, 2:4]
        return (self._eigvals_nonint, self._eigvecs_nonint,
                self._state_labels_nonint)

    def _spectrum_coupled(self):
        """Spectrum and level labels of the interacting system.

        Returns
        -------
        1d array
            Energies of the interacting system in ascending order.
        1d array of :class:`qutip.Qobj`
            Interacting eigenvectors.
        2d array
            State labels in the same order as energies.
            1st column - tuple-like labels such as (0, 1)
                for the state '01' of a two-qubit system.
            2nd column - string labels such as '01'.
        """
        if (self._eigvals is None or self._eigvecs is None
                or self._state_labels is None):
            eigvals, eigvecs = self.H().eigenstates()
            state_labels = np.empty((self._nlev, 2), dtype=object)
            _, eigvecs_nonint, labels_nonint = self._spectrum_nonint()
            for ind, vec in enumerate(eigvecs):
                # Find the noninteracting state corresponding to vec.
                indmax = np.argmax([np.abs(vec.overlap(eigvecs_nonint[k]))
                                    for k in range(self._nlev)])
                state_labels[ind, :] = labels_nonint[indmax, :]
                # Adjust the phase (sign) of the interacting state
                # to match the noninteracting one and
                # to keep the same sign among different machines.
                phase = np.angle(vec.overlap(eigvecs_nonint[indmax]))
                if np.abs(phase) > 1e-5:
                    if np.abs(phase - np.pi) < 1e-5:
                        eigvecs[ind] *= -1
                    else:
                        eigvecs[ind] *= np.exp(1j*phase)
            self._eigvals = eigvals
            self._eigvecs = eigvecs
            self._state_labels = state_labels
        return self._eigvals, self._eigvecs, self._state_labels

    def level_label(self, label, label_format='int', interaction='on'):
        """Converts a label of an energy level into a different format.

        Possible formats of a label:
            int - the sequential index of the level in ascending order
                of energies
            tuple - description of the corresponding noninteracting
                state consisting of indices describing the states of
                underlying objects such as (0, 1) for the state '01'
                of a two-qubit system
            str - string description of the corresponding noninteracting
                state such as '01' for a two-qubit system
        For 'tuple' and 'str' formats, the corresponding noninteracting
        states are chosen assuming the same order of energies, i.e.,
        ignoring the level crossings.

        Parameters
        ----------
        label : int, tuple, or str
        label_format : str (optional)
            Format of the return label ('int', 'tuple', or 'str')
        interaction : 'on' or 'off', optional

        Returns
        -------
        int, tuple, or str

        Example
        -------
        The ground state of the system with three objects.

        >>> level_label('000')  # returns 0
        >>> level_label(0, label_format='str') # returns '000'
        """
        if interaction == 'off':
            _, _, labels = self._spectrum_nonint()
        elif interaction == 'on':
            _, _, labels = self._spectrum_coupled()
        else:
            raise Exception('Unrecognized interaction flag')
        for k in range(len(labels)):
            if label == labels[k, 0] or label == labels[k, 1] or label == k:
                index = k
                break
            if k == len(labels) - 1:
                raise Exception('Unrecognized state label.')
        if label_format == 'int':
            return index
        elif label_format == 'tuple':
            return labels[index, 0]
        elif label_format == 'str':
            return labels[index, 1]
        else:
            raise Exception('Unrecognized format for the return label.')
            return None

    def levels(self, nlev=None, interaction='on', return_eigvecs=False):
        """Spectrum of the system.

        Parameters
        ----------
        nlev : int, optional
            The number of levels to return. Default is all the levels.
        interaction : 'on' or 'off', optional
            Return energy levels with or without coupling.
        return_eigvecs : bool, optional
            If True, return eigenvectors in addition to eigenvalues.

        Returns
        -------
        1d ndarray
            Energies in ascending order.
        1d array of :class:`qutip.Qobj` if `return_eigvecs` is True
            Eigenvectors corresponding to `eigenvalues`.
        """
        if interaction == 'on':
            spectrum_func = self._spectrum_coupled
        elif interaction == 'off':
            spectrum_func = self._spectrum_nonint
        else:
            raise Exception('Unrecognized interaction option.')
        eigvals, eigvecs, _ = spectrum_func()
        if nlev is None:
            nlev = self._nlev
        if return_eigvecs:
            return eigvals[:nlev], eigvecs[:nlev]
        else:
            return eigvals[:nlev]

    def level(self, label, interaction='on', return_eigvec=False):
        """Energy and eigenvector of a single level.

        Parameters
        ----------
        label : int, tuple, str
            Label of the level: sequential index of the level or its tuple
            or string description in terms of states of uncoupled objects
            such as (0, 1) or '01' for a two-qubit system.
        interaction : 'on' or 'off', optional
            Return eigenstate with or without coupling.
        return_eigvec : bool, optional
            If True, return eigenvector in addition to eigenvalue.

        Returns
        -------
        float
            Energy of the level.
        :class:`qutip.Qobj` if `return_eigvec` is True.
            Eigenvector.

        Examples
        --------
        >> level('01')  # Energy of the level labeled as '01'.
        """
        level_index = self.level_label(label, interaction=interaction)
        if return_eigvec:
            return_tuple = self.levels(
                interaction=interaction, return_eigvecs=True)
            return return_tuple[0][level_index], return_tuple[1][level_index]
        else:
            return self.levels(interaction=interaction)[level_index]

    def levels_nonint(self, nlev=None, return_eigvecs=False):
        """A shortcut for levels(interaction='off')."""
        return self.levels(
            nlev=nlev, interaction='off', return_eigvecs=return_eigvecs)

    def level_nonint(self, label, return_eigvec=False):
        """A shortcut for level(label, interaction='off')."""
        return self.level(
            label, interaction='off', return_eigvec=return_eigvec)

    def freq(self, level1, level2, interaction='on'):
        """Transition frequency defined as the energy difference.

        Parameters
        ----------
        level1, level2 : int, tuple, or str
            Level labels; e.g., '10' and '00' for 00-10 transition
            frequency.
        interaction : 'on' or 'off', optional
            Return the frequency of the interacting or uncoupled
            system.

        Returns
        -------
        float
            Transition frequency between `level1` and `level2` defined
            as the difference of energies.
            Positive if `level1` is below `level2` in energy.

        See Also
        --------
        level
        """
        return (self.level(level2, interaction=interaction)
                - self.level(level1, interaction=interaction))

    def freq_nonint(self, level1, level2):
        """A shortcut for freq(interaction='off')."""
        return self.freq(level1, level2, interaction='off')

    def eigvecs(self, nlev=None, interaction='on'):
        """A shortcut to get eigenvectors via levels().

        Returns
        -------
        1d array of :class:`qutip.Qobj`
            Eigenvectors.

        See Also
        --------
        levels
        """
        _, evecs = self.levels(
                nlev=nlev, interaction=interaction, return_eigvecs=True)
        return evecs

    def eigvec(self, label, interaction='on'):
        """A shortcut to get an eigenvector via level().

        Returns
        -------
        :class:`qutip.Qobj`
            Eigenvector.

        See Also
        --------
        level
        """
        _, evec = self.level(
            label, interaction=interaction, return_eigvec=True)
        return evec

    def eigvecs_nonint(self, nlev=None):
        """A shortcut for eigvecs(interaction='off')."""
        return self.eigvecs(nlev=nlev, interaction='off')

    def eigvec_nonint(self, label):
        """A shortcut for eigvec(label, interaction='off')."""
        return self.eigvec(label, interaction='off')

    def matr_el(self, obj, operator, level1, level2, interaction='on'):
        """Matrix element of an operator for a specific object."""
        operator = self.promote_op(obj, operator)
        evec1 = self.eigvec(level1, interaction=interaction)
        evec2 = self.eigvec(level2, interaction=interaction)
        return operator.matrix_element(evec1.dag(), evec2)

    def matr_el_nonint(self, obj, operator, level1, level2):
        """A shortcut for matr_el(interaction='off')."""
        return self.matr_el(obj, operator, level1, level2, interaction='off')

    def phi_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the flux operator."""
        return self.matr_el(
                obj, obj.phi(), level1, level2, interaction=interaction)

    def phi_ij_nonint(self, obj, level1, level2):
        """A shortcut for phi_ij(interaction='off')."""
        return self.phi_ij(obj, level1, level2, interaction='off')

    def n_ij(self, obj, level1, level2, interaction='on'):
        """Matrix element of the charge operator."""
        return self.matr_el(
                obj, obj.n(), level1, level2, interaction=interaction)

    def n_ij_nonint(self, obj, level1, level2):
        """A shortcut for n_ij(interaction='off')."""
        return self.n_ij(obj, level1, level2, interaciton='off')

    def projection(self, level_label, interaction='on'):
        """Projection operator for a system level."""
        psi = self.eigvec(level_label, interaction=interaction)
        return psi * psi.dag()

    def projection_nonint(self, level_label):
        """A shortcut for projection(interaction='off')."""
        return self.projection(level_label, interaction='off')
