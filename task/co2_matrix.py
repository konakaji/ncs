from qwrapper.hamiltonian import to_matrix_hamiltonian, compute_ground_state, Hamiltonian

hamiltonian = Hamiltonian.load("co2_hamiltonian.tsv")
# This already 
to_matrix_hamiltonian(hamiltonian)
compute_ground_state(hamiltonian)