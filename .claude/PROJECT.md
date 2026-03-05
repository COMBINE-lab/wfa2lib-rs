# Making a Rust-native port of WFA2-lib

This project is designed to be an *output-identical*, *performance-competitive*, *feature-complete* and *Rust-native* port of [WFA2-lib](https://github.com/smarco/WFA2-lib), a library for time and 
space-efficient sequence alignment.  

## Project Layout

There is a local version of the C/C++ WFA2-lib checked out here in `WFA2-lib`. We can use that local checkout for instrumetnation, and validation. 

### Testing

 - There is a comprehensive testing script for the C version located at `WFA2-lib/tests/wfa.utest.sh`.  We can create an appropriate Rust version of this by replacing the C-based driver binary with our Rust based driver binary.  This provides a set of known problems and solutions that we can use to evaluate our implementation, independent of the C/C++ implementation.
 - The C/C++ solution is the guide star. In any implementation question, we should defer to the choice made in the C/C++ code, and adopt the same strategy. Our output should be identical to the C/C++ solution. Thus, when tracking down differences and trying to fix bugs, we should feel free to trace a problem instance through the C/C++ solution, and through our solution to eliminate any differences.

### Design

 - The program already provides a very nice and usable C++ API. We should strive to provide a similar API. However, we want our API to be ergonomic and easy to use from Rust, so don't be afraid to propose potential improvements that will make the API nicer to use in Rust.
 - This is a *performance critical* library, therefore, while we care about an ergonomic API, we care even more about maintaining C-level performance. This means that we should think about allocations, how to avoid intializing memory that is written before it is read, and how to allow the user to keep around and propagate reusable "contexts" with temporary workspace, so that we can avoid frequent new allocations.
 - We should design our code as a library first, but should provide a binary driver program with the same interface as the canonical WFA2-lib
 - A main selling point of WFA2-lib is also that it is *memory efficient*. Therefore, when we evaluate our implemetnation against the baseline in benchmarking, we should care both about our runtime and our memory usage.

### Rust version

The Rust version should be rooted in this current top-level directory. We will have both a library, and a binary driver that uses that library.

 - Use Rust 2024 edition
 - Use MSRV 1.91
 - Use the latest versions of all crates
 - For logging use the tracing and tracing-subscriber crates
 - Feel free to pull in other crates, but ask me before doing so
 - For *local* development, have a `.cargo/config.toml` that sets `target-cpu=native`
 - Ensure that code lints cleanly with clippy, and make sure code is properly formatted before any commit.

## Project Tracking

We should track all of our progress in a file called `.claude/PROGRESS.md`. We should track any non-trivial bugs we find in a file called `.claude/FIXED_BUGS.md`.  We should maintain our current plan in a file called `.claude/PLAN.md`, and update it after every major milestone.  We should track our performance over time and with different specific code versions (tracked by git commit) in a file called `.claude/PERFORMANCE_LOG.md`, and refer to it to ensure that we don't introduce regressions.

