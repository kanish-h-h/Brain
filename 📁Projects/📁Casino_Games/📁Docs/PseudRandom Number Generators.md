### Notes on Pseudorandom Number Generators (PRNGs)

**Overview:**
- **PRNG (Pseudorandom Number Generator)**: An algorithm designed to generate a sequence of numbers that approximates the properties of random sequences. These numbers are not truly random but are determined by an initial seed.
- Truly random numbers are hard to generate, so pseudo-random numbers are used instead. 
- Pseudo-random numbers are obtained from a pseudo-random number generator (PRNG), which is supplied by the cryptographic library. 
- PRNGs need one starting value, called the seed, in order to generate pseudo-random numbers. Setting this initial value in the PRNG is known as `seeding the PRNG`.
- **Uses**: Simulations (e.g., Monte Carlo), gaming, cryptography.
- **Reproducibility**: PRNGs are valued for their speed and ability to reproduce results, as opposed to true random number generators.

**Key Concepts:**
- **Seed**: The initial value that starts the PRNG algorithm. Changing the seed changes the output sequence.
- **Predictability**: Cryptographic applications require PRNGs that prevent output predictability, even when earlier outputs are known.

**Potential Issues with PRNGs:**
- Common PRNGs may produce artifacts, such as:
  - **Shorter-than-expected periods**: For certain seed values, PRNGs may repeat sequences too quickly.
  - **Non-uniform distribution**: The generated numbers might not be evenly spread across the expected range.
  - **Correlated successive values**: Numbers generated in sequence may be related in undesirable ways.
  - **Dimensional distribution issues**: Poor coverage of output numbers in multi-dimensional space.
  - **Flawed algorithms**: Some, like RANDU, were used for decades despite being flawed.

**Notable Algorithms & Approaches:**
1. **Mersenne Twister (1997)**: 
   - Well-known for avoiding many issues found in earlier algorithms.
   - Period of 2^19937 - 1 iterations and can be distributed in up to 623 dimensions for 32-bit values.
2. **Xorshift Generators (2003)**: 
   - Extremely fast, based on linear recurrences.
3. **WELL Generators (2006)**: 
   - Improves upon the Mersenne Twister, addressing the large state space and slow recovery from zero-filled states.
4. **Early Method (Middle Square)**:
   - Proposed by John von Neumann in 1946. Though simple, it eventually repeats sequences and is now obsolete.

**Cryptographic PRNGs:**
- **CSPRNG** (Cryptographically Secure PRNG):
   - Designed for secure cryptographic use.
   - Must pass statistical tests and ensure that no previous or future values can be predicted from the current sequence.
   - Examples: CryptGenRandom (Microsoft), Yarrow (Mac OS X), and Fortuna.
   - Some algorithms like the Micali-Schnorr generator and Blum Blum Shub provide provable security but are slower.

**German BSI Evaluation Criteria for PRNGs:**
- **K1**: Sequences must be highly likely to differ from each other.
- **K2**: Output must pass statistical tests that check for randomness.
- **K3**: It should be infeasible for an attacker to predict future values based on past sequences.
- **K4**: The internal state of the generator should not be used to derive previous numbers.

**Mathematical Definition:**
- A PRNG is defined for probability distribution **P** over real numbers, following specific conditions for statistical equivalence with a random sequence.

**Generating Non-Uniform Random Numbers:**
- PRNGs can be used to generate numbers from a non-uniform distribution by applying a transformation to the uniform distribution.

**Random Number Generation**
In applications that rely heavily on random number generation, such as Monte Carlo simulations or stochastic processes, the choice between fixed and dynamic seeds depends on the specific requirements:
- Fixed seeds can be used when the same sequence of random numbers is required for a specific simulation or analysis.
- Dynamic seeds are suitable when a truly random sequence is necessary, such as in simulations involving complex systems or uncertainty analysis.