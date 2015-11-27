#ifndef MPI_NODE__
#define MPI_NODE__

#define MAX_NODE 100
#define MAX_PROCESS 200

struct process{
	int rank;
	int size;
};

struct node{
	char name[128];
	int id;
	int psize;
	int allocsize;
	struct process p[MAX_PROCESS];
};

struct node arraynode[MAX_NODE];
int PROCESS_SIZE;
int NODE_SIZE;
int process[MAX_PROCESS];
//初始化
int MPIN_init(char *path,int size);
//得到下一个进程的rank
int MPIN_get_next_rank(int curRank);
//得到上一个进程的rank
int MPIN_get_prev_rank(int curRank);
//得到总的节点数
int MPIN_get_node_size();
//根据Node节点号，得到当前节点进程规模
int MPIN_get_node_process_size(int nodeid);
//得到当前节点上的master rank
int MPIN_get_master_rank(int nodeid);
//根据node节点号和位置得到当前进程号
int MPIN_get_node_process_rank(int nodeid,int loc);
int MPIN_get_node_by_rank(int rank);
#endif
