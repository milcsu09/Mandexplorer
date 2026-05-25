#ifndef THREAD_POOL_H
#define THREAD_POOL_H


#include "common.h"


struct thread_pool;


struct thread_pool *
thread_pool_create (usize threads_amount, usize queue_capacity);

// NOTE: Waits for all currently queued tasks to finish.
void
thread_pool_destroy (struct thread_pool *pool);

void
thread_pool_clear (struct thread_pool *pool);

i32
thread_pool_enqueue (struct thread_pool *pool, void (*function) (void *), void *argument);

void
thread_pool_wait (struct thread_pool *pool);

bool
thread_pool_is_done (struct thread_pool *pool);

i32
thread_pool_threads_active (struct thread_pool *pool);


#endif // THREAD_POOL_H

