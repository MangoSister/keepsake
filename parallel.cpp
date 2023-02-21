#include "parallel.h"
#include <mutex>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

namespace ks
{

std::mutex mutex_init_parallel;
bool parallel_initialized = false;
std::unique_ptr<tbb::global_control> global_control;

inline int get_default_num_threads() { return tbb::this_task_arena::max_concurrency(); }

void init_parallel(int nthreads)
{
    // Make sure no race condition...
    std::scoped_lock lock(mutex_init_parallel);

    if (!parallel_initialized) {
        parallel_initialized = true;

        if (nthreads <= 0) {
            nthreads = get_default_num_threads();
        }
        global_control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, nthreads);
    }
}

} // namespace ks