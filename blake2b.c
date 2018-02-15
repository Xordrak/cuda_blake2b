#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include "blake2b.h"

// Endian Transforms
uint64_t load64(const void *src) {
#if defined(NATIVE_LITTLE_ENDIAN)
  uint64_t w;
  memcpy(&w, src, sizeof w);
  return w;
#else
  const uint8_t *p = (const uint8_t *)src;
  return ((uint64_t)(p[0]) <<  0) | ((uint64_t)(p[1]) <<  8) | ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40) | ((uint64_t)(p[6]) << 48) | ((uint64_t)(p[7]) << 56) ;
#endif
}

void store32(void *dst, uint32_t w) {
#if defined(NATIVE_LITTLE_ENDIAN)
  memcpy(dst, &w, sizeof w);
#else
  uint8_t *p = ( uint8_t * )dst;
  p[0] = (uint8_t)(w >>  0); p[1] = (uint8_t)(w >>  8); p[2] = (uint8_t)(w >> 16); p[3] = (uint8_t)(w >> 24);
#endif
}

// Blake Inits
void blake2b_init0(blake2b_state *S) {
  size_t i;
  memset(S, 0, sizeof(blake2b_state));

  for(i = 0; i < 8; ++i) S->h[i] = blake2b_IV[i];
}

/* init xors IV with input parameter block */
int blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
  const uint8_t *p = (const uint8_t *)(P);
  size_t i;

  blake2b_init0(S);

  /* IV XOR ParamBlock */
  for(i = 0; i < 8; ++i)
    S->h[i] ^= load64(p + sizeof( S->h[i] ) * i);

  S->outlen = P->digest_length;
  return 0;
}

int blake2b_init(blake2b_state *S, size_t outlen) {
  blake2b_param P[1];

  if ((!outlen) || (outlen > BLAKE2B_OUTBYTES)) return -1;

  P->digest_length = (uint8_t)outlen;
  P->key_length    = 0;
  P->fanout        = 1;
  P->depth         = 1;
  store32(&P->leaf_length, 0);
  store32(&P->node_offset, 0);
  store32(&P->xof_length, 0);
  P->node_depth    = 0;
  P->inner_length  = 0;
  memset(P->reserved, 0, sizeof(P->reserved));
  memset(P->salt,     0, sizeof(P->salt));
  memset(P->personal, 0, sizeof(P->personal));
  return blake2b_init_param(S, P);
}

