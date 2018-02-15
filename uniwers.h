#ifndef _COMMON_H
#define _COMMON_H

#define TRY(call)                                                            \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d \n", __FILE__, __LINE__);              \
        fprintf(stderr, "code: %d, reason: %s \n", error,                    \
                cudaGetErrorString(error));                                  \
        exit(1);                                                             \
    }                                                                        \
}

#endif // _COMMON_H

