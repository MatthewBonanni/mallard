// Runs a case through Mallard's C++ library API, as `mallard -i input.toml` does, reporting
// every 100 steps: ./run_case input.toml [--kokkos-num-threads=N]
#include <iostream>

#include <mallard/mallard.h>

int main(int argc, char ** argv) {
    // MPI and Kokkos for the whole program; they take their own options out of argv
    mallard::Session session(argc, argv);
    if (argc != 2) {
        std::cerr << "usage: run_case input.toml [--kokkos-num-threads=N]\n";
        return 2;
    }
    try {
        mallard::Solver solver(mallard::Case::from_toml_file(argv[1]));
        solver.setup();
        while (solver.step()) {
            if (solver.step_count() % 100 == 0) {
                std::cout << "run_case: step " << solver.step_count() << ", t = " << solver.time() << "\n";
            }
        }
        for (const mallard::Monitor & m : solver.monitors()) {
            std::cout << "run_case: " << m.name << " = " << m.value << " " << m.unit << "\n";
        }
        return solver.finish();
    } catch (const mallard::Error & e) {
        std::cerr << "error " << e.code() << ": " << e.what() << "\n";
        return 1;
    }
}
