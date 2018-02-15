# cuda_blake2b
Blake2b hash function for exec on NVIDIA GPU

Execution of BLAKE2B cryptographic hash function on the NVIDIA GPU.
Compile (CUDA SDK must be installed):

 -for Kepler-chipset:  nvcc -arch=sm_30 blake2b.c blake2b-gpu.cu -o gpu_b2b_sum
 
 -for Maxwell-chipset: nvcc -arch=sm_40 blake2b.c blake2b-gpu.cu -o gpu_b2b_sum
 
 -for Pascal-chipset:  nvcc -arch=sm_50 blake2b.c blake2b-gpu.cu -o gpu_b2b_sum

Common usage:
(path_to_)gpu_b2b_sum "Any text"
for example: ./gpu_b2b_sum "The quick brown fox jumps over the lazy dog"

Special usage:
(path_to_)gpu_b2b_sum "Any text" hash_len
where last parametr is non-standart length of resulting hash; it must be in range [10, 63); otherwise, it would be used standart hash-length = 64
for example: ./gpu_b2b_sum "The quick brown fox jumps over the lazy dog" 25
