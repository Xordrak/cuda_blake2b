# cuda_blake2b
Blake2b hash function for exec on NVIDIA GPU

Execution of BLAKE2B cryptographic hash function on the NVIDIA GPU.                                
Compile (CUDA SDK must be installed):                                                              
 select appropriate arch for your GPU-chipset and CUDA-version and run nvcc;                       
 for example, in case Ampere-chipset and cuda-12.6:                                                
     nvcc -arch=sm_86 blake2b.c blake2b-gpu.cu -o gpu_b2b_sum                                     
(nvcc-examples for old chipsets and old CUDA-versions can be viewed in branch snapshot-2018)       

Common usage:                                                                                      
(path_to_)gpu_b2b_sum "Any text"                                                                   
for example: ./gpu_b2b_sum "The quick brown fox jumps over the lazy dog"                           

Special usage:                                                                                     
(path_to_)gpu_b2b_sum "Any text" hash_len                                                          
where last parametr is non-standart length of resulting hash; it must be in range 9 < hash_len < 64; otherwise, it would be used standart hash-length = 64                                     
for example: ./gpu_b2b_sum "The quick brown fox jumps over the lazy dog" 25                        
