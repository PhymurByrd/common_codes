#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <stddef.h>

#ifndef MM_PRIVATE
#if defined(__GNUC__) || defined (__llvm__)
#define MM_PRIVATE __attribute__((visibility("hidden")))
#define MM_NONNULL __attribute__((nonnull))
#define MM_USERESULT __attribute__((warn_unused_result))
#else
#define MM_PRIVATE
#define MM_NONNULL
#define MM_USERESULT
#endif
#endif

struct mempool;
typedef struct mempool *mempoolptr;

MM_PRIVATE void* mempool_create(mempoolptr *mptr, const unsigned int size, unsigned int capacity, void* (*malloc)(size_t), void (*free)(void*));
MM_PRIVATE void* mempool_alloc(mempoolptr *mptr, const unsigned int size, const unsigned int capacity);
MM_PRIVATE void mempool_destroy(mempoolptr m);

#endif
