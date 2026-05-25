#include "thread_pool.h"
#include <SDL2/SDL_thread.h>
#include <SDL2/SDL_mutex.h>


struct thread_pool_work
{
  void (*function) (void *);

  void *argument;
};


struct thread_pool
{
  SDL_mutex *mutex;

  SDL_cond *cond_work;
  SDL_cond *cond_done;

  SDL_Thread **threads;

  usize threads_amount;
  usize threads_active;

  struct thread_pool_work *queue;

  usize queue_capacity;
  usize queue_size;
  usize queue_head;
  usize queue_tail;

  bool stop;
};


static int
thread_pool_thread_work (void *argument);


static void
thread_pool_work_enqueue (struct thread_pool *pool, struct thread_pool_work work)
{
  pool->queue_size++;

  pool->queue[pool->queue_tail] = work;

  pool->queue_tail = (pool->queue_tail + 1) % pool->queue_capacity;
}


static void
thread_pool_work_dequeue (struct thread_pool *pool, struct thread_pool_work *work)
{
  pool->queue_size--;

  *work = pool->queue[pool->queue_head];

  pool->queue_head = (pool->queue_head + 1) % pool->queue_capacity;
}


struct thread_pool *
thread_pool_create (usize threads_amount, usize queue_capacity)
{
  struct thread_pool *pool = malloc (sizeof (struct thread_pool));

  pool->mutex = SDL_CreateMutex ();

  pool->cond_work = SDL_CreateCond ();
  pool->cond_done = SDL_CreateCond ();

  pool->threads = calloc (threads_amount, sizeof (SDL_Thread *));

  pool->threads_amount = threads_amount;
  pool->threads_active = 0;

  pool->queue = calloc (queue_capacity, sizeof (struct thread_pool_work));

  pool->queue_capacity = queue_capacity;
  pool->queue_size = 0;
  pool->queue_head = 0;
  pool->queue_tail = 0;

  pool->stop = false;

  for (usize i = 0; i < threads_amount; ++i)
    pool->threads[i] = SDL_CreateThread (thread_pool_thread_work, "thread_pool_worker", pool);

  return pool;
}


void
thread_pool_destroy (struct thread_pool *pool)
{
  SDL_LockMutex (pool->mutex);

  pool->stop = true;

  SDL_CondBroadcast (pool->cond_work);

  SDL_UnlockMutex (pool->mutex);

  for (usize i = 0; i < pool->threads_amount; ++i)
    SDL_WaitThread (pool->threads[i], NULL);

  SDL_DestroyMutex (pool->mutex);

  SDL_DestroyCond (pool->cond_work);
  SDL_DestroyCond (pool->cond_done);

  free (pool->queue);

  free (pool->threads);

  free (pool);
}


void
thread_pool_clear (struct thread_pool *pool)
{
  SDL_LockMutex (pool->mutex);

  pool->queue_size = 0;
  pool->queue_head = 0;
  pool->queue_tail = 0;

  if (pool->threads_active == 0)
    SDL_CondSignal (pool->cond_done);

  SDL_UnlockMutex (pool->mutex);
}


i32
thread_pool_enqueue (struct thread_pool *pool, void (*function) (void *), void *argument)
{
  SDL_LockMutex (pool->mutex);

  if (pool->stop || pool->queue_size == pool->queue_capacity)
    {
      SDL_UnlockMutex (pool->mutex);
      return 1;
    }

  struct thread_pool_work work;

  work.function = function;
  work.argument = argument;

  thread_pool_work_enqueue (pool, work);

  SDL_CondSignal (pool->cond_work);

  SDL_UnlockMutex (pool->mutex);

  return 0;
}


void
thread_pool_wait (struct thread_pool *pool)
{
  SDL_LockMutex (pool->mutex);

  while (pool->queue_size > 0 || pool->threads_active > 0)
    SDL_CondWait (pool->cond_done, pool->mutex);

  SDL_UnlockMutex (pool->mutex);
}


bool
thread_pool_is_done (struct thread_pool *pool)
{
  SDL_LockMutex (pool->mutex);

  const bool done = pool->queue_size == 0 && pool->threads_active == 0;

  SDL_UnlockMutex (pool->mutex);

  return done;
}


i32
thread_pool_threads_active (struct thread_pool *pool)
{
  SDL_LockMutex (pool->mutex);

  const i32 active = pool->threads_active;

  SDL_UnlockMutex (pool->mutex);

  return active;
}


static int
thread_pool_thread_work (void *argument)
{
  struct thread_pool *pool = argument;

  while (1)
    {
      SDL_LockMutex (pool->mutex);

      while (!pool->stop && pool->queue_size == 0)
        SDL_CondWait (pool->cond_work, pool->mutex);

      if (pool->stop && pool->queue_size == 0)
        {
          SDL_UnlockMutex (pool->mutex);
          break;
        }

      struct thread_pool_work work;

      thread_pool_work_dequeue (pool, &work);

      pool->threads_active++;

      SDL_UnlockMutex (pool->mutex);

      if (work.function)
        work.function (work.argument);

      SDL_LockMutex (pool->mutex);

      pool->threads_active--;

      if (pool->queue_size == 0 && pool->threads_active == 0)
        SDL_CondSignal (pool->cond_done);

      SDL_UnlockMutex (pool->mutex);
    }

  return 0;
}

