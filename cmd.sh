# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export big_memory_cmd="queue.pl -l arch=*64,ram_free=8G,mem_free=8G"
#export cuda_cmd="queue.pl -l gpu=1"

#c) run it locally... works for CMU rocks cluster



export train_cmd="slurm.pl"
export huge_mem="slurm.pl --mem 150G"
export large_mem="slurm.pl --mem 20G"
export cpu_cmd="slurm.pl -x cqxx-01-006 --num_threads 5"
export decode_cmd="slurm.pl -p 2080ti,gpu -x gqxx-01-116,gqxx-01-114,gqxx-01-096,gqxx-01-160,gqxx-01-0[66-70],gqxx-01-[117-123],gqxx-01-131,gqxx-01-141,gqxx-01-145,gqxx-01-019,gqxx-01-027,gqxx-01-086 --gpu 1 --mem 15G --num-threads 3"
export cuda_cmd="slurm.pl -p 2080ti,gpu -x gqxx-01-116,gqxx-01-114,gqxx-01-096,gqxx-01-160,gqxx-01-0[66-70],gqxx-01-[117-123],gqxx-01-131,gqxx-01-141,gqxx-01-145,gqxx-01-019,gqxx-01-027,gqxx-01-086 --gpu 1 --mem 15G --num-threads 3"
export local_cmd="run.pl"
export cpu_cmd="run.pl"
#export cuda_cmd="run.pl"
export cuda_cmd="slurm.pl -x gqxx-01-134,gqxx-01-145,gqxx-01-104,gqxx-01-056 -p 2080ti,gpu --gpu 1 --mem 5G"
export decode_cmd="slurm.pl -p 2080ti,gpu --gpu 1 --mem 15G"
export decode_cmd="slurm.pl"
#export cpu_cmd="slurm.pl -p 2080ti,gpu --gpu 1 --mem 15G"
export cpu_cmd="slurm.pl"
