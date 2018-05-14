/*
 ============================================================================
 Name        : triangleCountingDP.cu
 Author      : Nishith Tirpankar
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */


//#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <iterator>
#include <cmath>

#include <unistd.h>
#include "CSVReader.h"
#include "utils.h"
#include "graph.h"

#include <cuda.h>

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

using namespace cub;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


template<typename Key, int NUM_BUCKETS>
__device__ __forceinline__ int bucketnum(Key num, unsigned int bitnum){
	const unsigned int bits_in_buckets = Log2<NUM_BUCKETS>::VALUE;
	return (num >> (bitnum-bits_in_buckets+1)) & (NUM_BUCKETS-1);
}

template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void countKernel(
		Key 			*d_in,			// Tile of input
		unsigned int	*d_scan_in,		// Device scan location of size NUM_BUCKETS*NUM_THREADS where local count will be written
		unsigned int 	num_elems,		// Total number of elements to be counted
		unsigned int 	bitnum
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;
	int num_threads = blockDim.x * gridDim.x;
	Key l_cnt[NUM_BUCKETS] = {0};

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
		l_cnt[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]++;
	}

	// copy local count into global memory
	for(unsigned int i = 0; i < NUM_BUCKETS; i++){
		d_scan_in[i*num_threads+idx] = l_cnt[i];
	}
}

template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void copyKernel(
		Key				*d_in,		// Tile of input
		Key				*d_out,		// Tile of output
		unsigned int 	num_elems	// Total number of elements to be moved
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	unsigned int block_offset = blockIdx.x * TILE_SIZE;
	unsigned int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
		d_out[i] = d_in[i];
	}
}

template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void moveKernel(
		Key 			*d_in,		// Tile of input
		Key 			*d_out,	 	// Tile of output
		unsigned int	*d_scan_out,// Device scan location of size NUM_BUCKETS*NUM_THREADS where local count will be written
		unsigned int 	num_elems,		// Total number of elements to be counted
		unsigned int 	bitnum
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;
	int num_threads = blockDim.x * gridDim.x;
	Key l_scan_inc[NUM_BUCKETS] = {0};

	for(unsigned int i = 0; i < NUM_BUCKETS; i++){
		l_scan_inc[i] = d_scan_out[i*num_threads+idx];
	}

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
		l_scan_inc[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]--;
		d_out[l_scan_inc[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]] = d_in[i];
	}
}

#define LOCAL_SORT_THRESHOLD 1024*8

template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    int			NUM_BUCKETS>
__global__ void radixSort(
    Key           *d_in,		// Tile of input
    Key           *d_out,		// Tile of buffer - this is where the output should be at a particular level/depth finally after the operation is complete
    unsigned int  *num_elems,	// The total number of elements in the list at this level - array of size NUM_BUCKETS
    unsigned int  *offsets,		// The offset from d_in where the current bucket starts - array of size NUM_BUCKETS
    unsigned int  *bitnum_beg	// the bit from which the buckets will be counted
    )
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	unsigned int l_offset = offsets[threadIdx.x];
	unsigned int l_num_elems = num_elems[threadIdx.x];
	Key *l_in = &d_in[l_offset];
	Key *l_out = &d_out[l_offset];
	unsigned int GRID_SIZE = (unsigned int)ceilf(((float)l_num_elems)/((float)TILE_SIZE));
	unsigned int NUM_THREADS = GRID_SIZE*BLOCK_THREADS;
	unsigned int *d_scan_in, *d_scan_out;
	unsigned int *d_next_num_elems, *d_next_offsets, *d_next_bitnum_beg;
	cudaError_t err;

	// Storage pointers for cubs
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
#if 0
	printf("STARTING radixsort for bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
#endif

	if(l_num_elems == 0){
#if 0
		printf("EXITING bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
#endif
		return;
	}
	// --------------------------------------------------------------------------------------------------------
	// if the number of elements in this bucket is too small or the next recursion will go below the zeroth bit
	if(l_num_elems <= LOCAL_SORT_THRESHOLD || ((((int)bitnum_beg[0]-(int)Log2<NUM_BUCKETS>::VALUE))< 0) ) {
		printf("LOCAL SORTING for bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
		DoubleBuffer<Key> d_keys(l_in, l_out);
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, l_num_elems);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		if(d_temp_storage == NULL){ printf("ERROR: d_temp_storage cudaMalloc failed.\n"); return;}
		DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, l_num_elems);
		cudaDeviceSynchronize();
		//d_out = d_keys.Current();
		if(d_keys.Current() != l_out){
			copyKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, l_num_elems);
			cudaDeviceSynchronize();
		}
		cudaFree((void *) d_temp_storage);
#if 0
		printf("CHECKING MEMORY d_out for l_num_elems %d:\n", l_num_elems);
		for(int i = 0; i < l_num_elems; i++){
			unsigned int x, y;
			x = *(unsigned int *)&(d_out[i]);
			y = *((unsigned int *)&(d_out[i]) + 1);
			printf("(%d, %d) ", y, x);
		}
		printf("\n");
#endif
		return;
	}

	// --------------------------------------------------------------------------------------------------------
	// Count
	//printf("Running count kernel now....\n");
	cudaMalloc((void ** )&d_scan_in, sizeof(unsigned int)*NUM_THREADS*NUM_BUCKETS);
	if(d_scan_in == NULL){ printf("ERROR: d_scan_in cudaMalloc failed.\n"); return;}
	countKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, d_scan_in, l_num_elems, bitnum_beg[0]);

	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: countKernel failed due to err code %d.\n", err); return;}

	cudaDeviceSynchronize();

	// prefix inclusive sum scan
	cudaMalloc((void ** )&d_scan_out, sizeof(unsigned int)*NUM_THREADS*NUM_BUCKETS);
	if(d_scan_out == NULL){ printf("ERROR: d_scan_out cudaMalloc failed.\n"); return;}
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_scan_in, d_scan_out, NUM_THREADS*NUM_BUCKETS);
//	printf("Scan temp_storage %d\n", temp_storage_bytes);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	if(d_temp_storage == NULL){ printf("ERROR: d_temp_storage cudaMalloc failed.\n"); return;}
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_scan_in, d_scan_out, NUM_THREADS*NUM_BUCKETS);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: DeviceScan::InclusiveSum failed due to err code %d.\n", err); return;}

	cudaDeviceSynchronize();
	cudaFree((void *) d_temp_storage);

	// --------------------------------------------------------------------------------------------------------
	// Bucket/move
	moveKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, d_scan_out, l_num_elems, bitnum_beg[0]);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: moveKernel failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();

	// --------------------------------------------------------------------------------------------------------
	// Recurse
	unsigned int l_bucket_scan[NUM_BUCKETS] = {0};									// for the buckets get the last element of d_scan_out
	cudaMalloc((void ** )&d_next_num_elems, sizeof(unsigned int)*NUM_BUCKETS);		// to get this for each value in l_bucket_scan subtract the previous value from it except for the first
	cudaMalloc((void ** )&d_next_offsets, sizeof(unsigned int)*NUM_BUCKETS); 		// for this convert the l_bucket_scan from inclusive to exclusive
	cudaMalloc((void ** )&d_next_bitnum_beg, sizeof(unsigned int));					// reduce the bitnum_beg by log2<num_buckets>:value-1

	if(d_next_num_elems == NULL){ printf("ERROR: d_next_num_elems cudaMalloc failed.\n"); return;}
	if(d_next_offsets == NULL){ printf("ERROR: d_next_offsets cudaMalloc failed.\n"); return;}
	if(d_next_bitnum_beg == NULL){ printf("ERROR: d_next_bitnum_beg cudaMalloc failed.\n"); return;}

	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: mallocs for d_next_* failed due to err code %d.\n", err); return;}

	for(int j = 0; j < NUM_BUCKETS; j++){
		l_bucket_scan[j] = d_scan_out[j*NUM_THREADS+(NUM_THREADS-1)];
		if(j == 0){
			d_next_num_elems[j] = l_bucket_scan[j];
			d_next_offsets[j] = 0;
		}else{
			d_next_num_elems[j] = l_bucket_scan[j] - l_bucket_scan[j-1];
			d_next_offsets[j] = l_bucket_scan[j-1];
		}
	}
	d_next_bitnum_beg[0] = bitnum_beg[0] - Log2<NUM_BUCKETS>::VALUE;

	// Free all but the next element arrays
	cudaFree((void *) d_scan_in);
	cudaFree((void *) d_scan_out);

	// actually recurse by calling multiple kernels - the l_in and l_out buffers ought to be reversed for the next iteration
	radixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<1, NUM_BUCKETS>>>(l_out, l_in, d_next_num_elems, d_next_offsets, d_next_bitnum_beg);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: radixSort failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();
	// The result will be l_in now. It needs to be moved back into l_out
	copyKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, l_num_elems);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: copyKernel failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();
	return;
}

//UniqueUblockCntKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_out, g_scan);
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void UniqueUblockCntKernel(
    Key          *d_in,          // Tile of input
    unsigned int *d_scan)        // single element of array for g_scan
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	// TODO: Check if the above kernel takes the result of the blocks and sorts it back - it does not - so use deviceradixsort
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x*ITEMS_PER_THREAD;
	int u_block_cnt = 0;

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD; i++)
	{
		// only for 0'th thread 0'th block 0th element do not check the prior value since its not present simply increment
		if(i == 0){
			u_block_cnt++;
			continue;
		}
		if((unsigned int)(d_in[i]>>(sizeof(unsigned int)*8)) != (unsigned int)(d_in[i-1]>>(sizeof(unsigned int)*8)))
			u_block_cnt++;
	}

	// write out to device memory
	int scanIdx = blockIdx.x*blockDim.x + threadIdx.x;
	d_scan[scanIdx] = u_block_cnt;
}

//UniqueUblockIndexKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_out, d_scan, d_index, d_u_block, E_prime_size);
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void UniqueUblockIndexKernel(
    Key          *d_in,         // Tile of sorted input list
    unsigned int *d_scan,       // the scan values that are passed into each thread
    unsigned int *d_index,      // the indexes of each u_block values
    unsigned int *d_u_block,    // single element of array for g_scan
    unsigned int *E_prime_size)     // The number of vertices in E' that it will produce (from degree of the vertex)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x*ITEMS_PER_THREAD;
	int scanIdx = blockIdx.x*blockDim.x + threadIdx.x;
	// read d_scan into local register
	unsigned int scan_offset = d_scan[scanIdx];

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD; i++)
	{
		// TODO: First allocate a local memory of size of d_cnt[scanIdx] and then move data to it one at a time
		// At the end of this loop copy the d_cnt[scanIdx] entries into device memory
		// only for 0'th thread 0'th block 0th element do not check the prior value since its not present simply increment
		if(i == 0){
			d_index[scan_offset] = i;
			d_u_block[scan_offset] = (unsigned int)(d_in[i]>>(sizeof(unsigned int)*8));
			scan_offset++;
			continue;
		}
		if((unsigned int)(d_in[i]>>(sizeof(unsigned int)*8)) != (unsigned int)(d_in[i-1]>>(sizeof(unsigned int)*8))){
			d_index[scan_offset] = i;
			d_u_block[scan_offset] = (unsigned int)(d_in[i]>>(sizeof(unsigned int)*8));
			scan_offset++;
		}
	}
	// To make sure that the last value is present in each threads write bound d_index
	__syncthreads();
	for(unsigned int i = d_scan[scanIdx]; i < scan_offset; i++){
		if(i != 0){
			unsigned int degree = d_index[i]-d_index[i-1];
			E_prime_size[i-1] = degree*(degree-1)/2;
		}
	}
}



//DEIndexPopulateKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<e_prime_gen_grid_size, BLOCK_THREADS>>>(d_E_prime_size_scan, d_e_index, h_scan_end+h_cnt_end);
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void DEIndexPopulateKernel(
    unsigned int *d_E_prime_size_scan,	// the inclusive scan of the e_prime_size array
    int *d_e_index,      		// the reference to the index array locations split up by the work
    unsigned int E_prime_size_scan_len)	// The number of elements in the d_E_prime_size_scan array
{
	// will optimize this later if possible
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x*ITEMS_PER_THREAD;
	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD; i++)
	{
		// If this thread is looking outside the bounds of d_E_prime_size_scan break out
		if(i >= E_prime_size_scan_len)
			break;
		if(i == 0){
			d_e_index[i] = 0;
			continue;
		}
		unsigned int prev = d_E_prime_size_scan[i-1]/ITEMS_PER_THREAD;
		unsigned int curr = d_E_prime_size_scan[i]/ITEMS_PER_THREAD;
		if(prev != curr){
			d_e_index[prev] = i;
			if(prev == 0)
				d_e_index[prev] = 0;
		}

	}
}

//EPrimeComputeKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<e_prime_gen_grid_size, BLOCK_THREADS>>>(d_index, d_e_index, d_E_prime_size_scan, h_scan_end+h_cnt_end, d_out, total_num_items, d_E_prime);
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void EPrimeComputeKernel(
    unsigned int *d_index,		// the
    int *d_e_index,      		// the reference to the index array locations split up by the work
    unsigned int *d_E_prime_size_scan,	// the E_prime_size_scan array inclusive
    unsigned int E_prime_size_scan_len,	// The number of elements in the d_E_prime_size_scan array
    Key *d_out,				// sorted E array
    int num_edges_E,			// number of edges in E which is the size of d_out
    Key *d_E_prime)			// output E_prime array    int *d_low, int *d_high)
{
	int thread_offset = blockIdx.x*blockDim.x + threadIdx.x;

	if(d_e_index[thread_offset] == -1)
		return;
	unsigned int index_low = d_e_index[thread_offset];
	unsigned int index_high = 0; // upper limit exclusive
	for(unsigned int i = thread_offset+1; i <= BLOCK_THREADS*gridDim.x; i++){
		if(i >= BLOCK_THREADS*gridDim.x){
			index_high = E_prime_size_scan_len;
			break;
		}
		if(d_e_index[i] != -1){
			index_high = d_e_index[i];
			break;
		}
	}
	//d_low[thread_offset] = index_low;
	//d_high[thread_offset] = index_high;
	// In the d_index array from index_low to < index_high compute all combinations
	for(unsigned int u_ptr = index_low; u_ptr < index_high; u_ptr++){
		unsigned int u_low = d_index[u_ptr];
		// The size of the original edge list E especially for the last one
		unsigned int u_high;
		if(u_ptr+1 >= E_prime_size_scan_len)
			u_high = num_edges_E;
		else
			u_high = d_index[u_ptr+1];
		unsigned int u_write_low;
		if(u_ptr == 0)
			u_write_low = 0;
		else
			u_write_low = d_E_prime_size_scan[u_ptr-1];
		//unsigned int u_write_high = d_E_prime_size_scan[u_ptr];
		unsigned int u_write_ptr = u_write_low;
		for(unsigned int i = u_low; i < u_high; i++){
			//for(unsigned int j = u_low+1; j < u_high; j++){
			for(unsigned int j = i+1; j < u_high; j++){
				edge<unsigned int> *temp = (edge<unsigned int> *)(d_E_prime+u_write_ptr);
				// assert if u_write_ptr > u_write_high - not implemented
				//d_low[u_write_ptr] = i;
				//d_high[u_write_ptr] = j;
				temp->u = ((edge<unsigned int> *)(&d_out[i]))->v;
				temp->v = ((edge<unsigned int> *)(&d_out[j]))->v;
				u_write_ptr++;
			}
		}
	}
	__syncthreads();
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
std::vector<edge<unsigned int>> deployKernel(unsigned long *h_in, int total_num_items){
	const int TILE_SIZE = BLOCK_THREADS*ITEMS_PER_THREAD;
	int g_grid_size = total_num_items/TILE_SIZE + 1;

	std::cout<<"g_grid_size: "<<g_grid_size<<" Tilesize: "<<TILE_SIZE<<" Total num items: "<<total_num_items<<std::endl;
	std::cout<<"BLOCK_THREADS: "<<BLOCK_THREADS<<" ITEMS_PER_THREAD: "<<ITEMS_PER_THREAD<<std::endl;

	unsigned long *d_in       = NULL;
	unsigned long *d_out      = NULL;
	unsigned long *d_key_alt_buf = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(unsigned long) * TILE_SIZE * g_grid_size));
	CubDebugExit(cudaMalloc((void**)&d_key_alt_buf, sizeof(unsigned long) * TILE_SIZE * g_grid_size));
	cudaError_t err;

	edge<unsigned int> *temp = new edge<unsigned int>(UINT_MAX, UINT_MAX);
	unsigned long *t = (unsigned long *)temp;

	// Set the last tile elements to max_keysize since it will be partially filled
	CubDebugExit(cudaMemset(d_in+(TILE_SIZE*(g_grid_size-1)), *t, sizeof(unsigned long) * TILE_SIZE));

#if 0
	edge<unsigned int> *p;
	p = (edge<unsigned int> *)h_in;
	for(int i = 0; i < total_num_items; i++){
		p[i].print();
		std::cout<<" ";
	}
	std::cout<<std::endl;
#endif

	// copy to device
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(unsigned long) * total_num_items, cudaMemcpyHostToDevice));

	// run kernel
	// This kernel sorts arrays in individual blocks - not as a whole - it can be used as a basis for block based computation
	// This uses device radixsort
	DoubleBuffer<unsigned long> d_keys(d_in, d_key_alt_buf);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	//DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, total_num_items);
	// TODO: Do we include the padding into the sort to replace the above line???
	DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, TILE_SIZE*g_grid_size);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	//DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, total_num_items);
	DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, TILE_SIZE*g_grid_size);
	d_out = d_keys.Current();

#if 0
	std::cout<<"Copying sorted data to h_in from d_out."<<std::endl;
	// copy data back to host
	CubDebugExit(cudaMemcpy(h_in, d_out, sizeof(unsigned long) * total_num_items, cudaMemcpyDeviceToHost));
	std::vector<edge<unsigned int>> p((edge<unsigned int> *)h_in, (edge<unsigned int> *)h_in+total_num_items);
	std::cout<<"The sorted array: "<<std::endl;
	for(int i = 0; i < total_num_items; i++){
		p[i].print();
		std::cout<<std::endl;
	}
#endif


	// DO DEGREE COUNTING HERE FOR NOW - LATER MOVE IT INTO A DIFFERENT FUNCTION AS REQUIRED
	// ------------------------------------------------------------------------------
	// 1. Divide sorted E into equal sized chunks [1 1 1 2 | 2 3 3 3 | 4 4 5 6 | 6 7 8 8]
	// 2. Find number of unique u's in your chunk [1 0 0 1 | 0 1 0 0 | 1 0 1 1 | 0 1 1 0]
	//    for i in mychunk
	//       if(E[i].u!=E[i-1].u)
	//          cnt++                             [    2   |     1   |     3   |     2  ]
	// 3. d_scan <- exclusive_scan(cnt)           [    0   |     2   |     3   |     6  |   8  ]
	// 4. Allocate array of size scan[n+1] where n+1 = scan[n] + cnt[last_thread] called index and u_block
	//             0 1 2 3 4 5 6 7
	//    index   [0 0 0 0 0 0 0 0]
	//    u_block [0 0 0 0 0 0 0 0]
	//    degree  [0 0 0 0 0 0 0 0]
	// 5. Find the degrees of the elements along with their offsets
	//    for i in mychunk
	//       if(E[i].u!=E[i-1].u){
	//          index[d_scan[tid]] = i
	//          u[d_scan[tid]] = E[i].u
	//          degree[d_scan[tid]-1] = i-index[d_scan[tid]-1]
	//          d_scan[tid]++
	//       }
	//             0 1 2 3  4  5  6  7
	//    index   [0 3 5 8 10 11 13 14]
	//    u_block [1 2 3 4  5  6  7  8]
	//    degree  [3 2 3 2  1  2  1  0]

	std::cout<<"1,2. Finding number of unique u_blocks in your chunk. "<<std::endl;
	// Allocate device memory for d_cnt
	unsigned int *d_cnt = NULL;
	CubDebugExit(cudaMalloc((void**)&d_cnt, sizeof(unsigned int) * BLOCK_THREADS * g_grid_size));
	// compute input for d_scan
	UniqueUblockCntKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_out, d_cnt);

	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: UniqueUblockCntKernel failed due to err code %d.\n", err); return std::vector<edge<unsigned int>>();}

#if 0
	unsigned int *h_scan;
	h_scan = new unsigned int[BLOCK_THREADS*g_grid_size]();
	// debug copy stuff to host - check result of UniqueUblockCnt
	CubDebugExit(cudaMemcpy(h_scan, d_cnt, sizeof(unsigned int) * BLOCK_THREADS * g_grid_size, cudaMemcpyDeviceToHost));
	std::cout<<"The count in each block is:"<<std::endl;
	for(int i = 0; i < BLOCK_THREADS*g_grid_size; i++){
		std::cout<<h_scan[i]<<" ";
	}
	std::cout<<std::endl;
#endif

	std::cout<<"3. Performing Exclusive Sum Scan on d_cnt. "<<std::endl;
	// perform device_exclusive scan sum
	unsigned int *d_scan = NULL;
	CubDebugExit(cudaMalloc((void**)&d_scan, sizeof(unsigned int) * BLOCK_THREADS * g_grid_size));

	// Allocate temporary storage for device scan
	//void            *d_temp_storage = NULL;
	//size_t          temp_storage_bytes = 0;
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cnt, d_scan, BLOCK_THREADS*g_grid_size));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	// Run
	CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cnt, d_scan, BLOCK_THREADS*g_grid_size));
	// Free intermediate memory d_temp_storage
	if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#if 0
	CubDebugExit(cudaMemcpy(h_scan, d_scan, sizeof(unsigned int) * BLOCK_THREADS * g_grid_size, cudaMemcpyDeviceToHost));
	std::cout<<"The scan of each block is:"<<std::endl;
	for(int i = 0; i < BLOCK_THREADS*g_grid_size; i++){
		std::cout<<h_scan[i]<<" ";
	}
	std::cout<<std::endl;

#endif
	std::cout<<"4. Allocating arrays for the index, u_block_vertices and the degrees of each. "<<std::endl;
	// Allocate array of size d_scan[-1]+d_cnt[-1] - its the count of unique vertices including the last MAX_INT vert if present
	unsigned int *d_index = NULL;
	unsigned int *d_u_block = NULL;
	unsigned int *d_E_prime_size = NULL;
	unsigned int h_scan_end, h_cnt_end;
	// to compute the array size we need to copy over the data from the device
	CubDebugExit(cudaMemcpy(&h_scan_end, &d_scan[BLOCK_THREADS*g_grid_size-1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(&h_cnt_end, &d_cnt[BLOCK_THREADS*g_grid_size-1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMalloc((void**)&d_index, sizeof(unsigned int) * (h_scan_end+h_cnt_end)));
	CubDebugExit(cudaMalloc((void**)&d_u_block, sizeof(unsigned int) * (h_scan_end+h_cnt_end)));
	CubDebugExit(cudaMalloc((void**)&d_E_prime_size, sizeof(unsigned int) * (h_scan_end+h_cnt_end)));

	// Find the offsets and u_block_element values of elements
	UniqueUblockIndexKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(d_out, d_scan, d_index, d_u_block, d_E_prime_size);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: UniqueUblockIndexKernel failed due to err code %d.\n", err); return std::vector<edge<unsigned int>>();}

	if (d_cnt) CubDebugExit(cudaFree(d_cnt));
	if (d_scan) CubDebugExit(cudaFree(d_scan));
#if 0
	unsigned int *h_index;
	unsigned int *h_u_block;
	unsigned int *h_degree;
	h_index = new unsigned int[h_scan_end+h_cnt_end];
	h_u_block = new unsigned int[h_scan_end+h_cnt_end];
	h_degree = new unsigned int[h_scan_end+h_cnt_end];
	// debug copy stuff to host - check result of UniqueUblockIndexKernel
	CubDebugExit(cudaMemcpy(h_index, d_index, sizeof(unsigned int) * (h_scan_end+h_cnt_end), cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(h_u_block, d_u_block, sizeof(unsigned int) * (h_scan_end+h_cnt_end), cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(h_degree, d_E_prime_size, sizeof(unsigned int) * (h_scan_end+h_cnt_end), cudaMemcpyDeviceToHost));
	std::cout<<"The index array is as follows: "<<std::endl;
	for(int i = 0; i < h_scan_end+h_cnt_end; i++){
		std::cout<<h_index[i]<<" ";
	}
	std::cout<<std::endl;
	std::cout<<"The u_block vertices array is as follows: "<<std::endl;
	for(int i = 0; i < h_scan_end+h_cnt_end; i++){
		std::cout<<h_u_block[i]<<" ";
	}
	std::cout<<std::endl;
	std::cout<<"The E_prime_size array is as follows: "<<std::endl;
	for(int i = 0; i < h_scan_end+h_cnt_end; i++){
		std::cout<<h_degree[i]<<" ";
	}
	std::cout<<std::endl;
#endif
	// ------------------------------------------------------------------------------


	// ------------------------------------------------------------------------------
	// Find and allocate the memory required to produce all the n(n-1)/2 combinations
	// 1. Find out the number of combinations each vertex will be responsible for producing
	//    This can be done in the prior step
	//             		 0 1 2 3  4  5  6  7
	//    index   		[0 3 5 8 10 11 13 14]
	//    u_block 		[1 2 3 4  5  6  7  8]
	//    degree  		[3 2 3 2  1  2  1  0]
	//    E_prime_size  	[3 1 3 1  0  1  0  0]
	// 2. Scan sum the size array inclusive
	//    E_prime_size_scan	[3 4 7 8  8  9  9  9]
	// 3. Create a new grid based on the number of elements in the E_prime_scan array
	//    e_prime_grid = |E_prime_size_scan|/(BLOCK_THREADS*ITEMS_PER_THREAD) + 1
	// 4. Allocate device memory for e_index which indicates where the combo_creation kernel will read
	//    We need to find |e_index| = BLOCK_THREADS*e_prime_gen_grid i.e. we need to find e_prime_gen_grid
	//    tile_size = BLOCK_THREADS*ITEMS_PER_THREAD = 16*2 = 32 (assume BLOCK_THREADS=16, ITEMS_PER_THREAD=2)
	//    e_prime_gen_grid = E_prime_size_scan[-1]/tile_size+1 = 9/32 + 1 = 0+1 = 1
	//    Thus, |e_index| = BLOCK_THREADS*e_prime_gen_grid = 16*1 = 16
	//    e_index 		[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	// 5. Launch a kernel with <<<e_prime_grid, BLOCK_THREADS>>> for step 3 which will write to e_index the read offsets
	//    for i in mychunk:
	//       if (E_prime_size_scan[i-1]/ITEMS_PER_THREAD) != (E_prime_size_scan[i]/ITEMS_PER_THREAD)
	//          e_index[E_prime_size_scan[i-1]/ITEMS_PER_THREAD] = i
	//          if E_prime_size_scan[i-1]/ITEMS_PER_THREAD == 0:
	//             e_index[E_prime_size[i-1]/ITEMS_PER_THREAD] = 0   // makes sure that thread 0 later reads from elem 0 on
	//          continue
	//       if i == 0:
	//          e_index[i] = 0      // makes sure that if there is no 0-1 transition in floor(E_prime_scan_size[i]/ITEMS_PER_THREAD) from the i-1 element the 0'th index is still set

	unsigned int *d_E_prime_size_scan = NULL;
	CubDebugExit(cudaMalloc((void**)&d_E_prime_size_scan, sizeof(unsigned int) * (h_scan_end+h_cnt_end)));
	// Scan sum the eprime
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_E_prime_size, d_E_prime_size_scan, h_scan_end+h_cnt_end));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	// Run
	CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_E_prime_size, d_E_prime_size_scan, h_scan_end+h_cnt_end));
	// Free intermediate memory d_temp_storage
	if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

#if 0
	unsigned int *h_E_prime_size_scan;
	h_E_prime_size_scan = new unsigned int[h_scan_end+h_cnt_end];
	// debug copy stuff to host - check result of InclusiveSum
	CubDebugExit(cudaMemcpy(h_E_prime_size_scan, d_E_prime_size_scan, sizeof(unsigned int) * (h_scan_end+h_cnt_end), cudaMemcpyDeviceToHost));
	std::cout<<"The E prime size scan array of size "<<h_scan_end+h_cnt_end<<" is: "<<std::endl;
	for(int i = 0; i < h_scan_end+h_cnt_end; i++){
		std::cout<<h_E_prime_size_scan[i]<<" ";
	}
	std::cout<<std::endl;
#endif

	// Allocate memory for e_index which indicates the locations in the d_index array that each thread will read
	// For this first compute the e_prime_gen_grid_size
	// Get the last element of d_E_prime_size_scan which is the size of e_index
	unsigned int h_e_prime_scan_end;
	// to compute the array size we need to copy over the data from the device
	CubDebugExit(cudaMemcpy(&h_e_prime_scan_end, &d_E_prime_size_scan[h_scan_end+h_cnt_end-1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int e_prime_gen_grid_size = h_e_prime_scan_end/TILE_SIZE + 1;
	std::cout<<"Allocating memory of size "<<BLOCK_THREADS*e_prime_gen_grid_size<<" for the d_e_index..."<<std::endl;
	std::cout<<"h_e_prime_scan_end = "<<h_e_prime_scan_end<<" e_prime_gen_grid_size = "<<e_prime_gen_grid_size<<std::endl;
	int *d_e_index = NULL;
	CubDebugExit(cudaMalloc((void**)&d_e_index, sizeof(int) * (BLOCK_THREADS*e_prime_gen_grid_size)));
	CubDebugExit(cudaMemset(d_e_index, -1, sizeof(int) * (BLOCK_THREADS* e_prime_gen_grid_size)));

	// With a new grid e_prime_grid on the array E_prime_size_scan populate the d_e_index array
	unsigned int e_prime_grid = (h_scan_end+h_cnt_end)/TILE_SIZE + 1;
	std::cout<<"Calling DEIndexPopulateKernel with e_prime_grid = "<<e_prime_grid<<" E_prime_size_scan_len = "<<h_scan_end+h_cnt_end<<std::endl;
	DEIndexPopulateKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<e_prime_grid, BLOCK_THREADS>>>(d_E_prime_size_scan, d_e_index, h_scan_end+h_cnt_end);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: DEIndexPopulateKernel failed due to err code %d.\n", err); return std::vector<edge<unsigned int>>();}

#if 0
	int *h_e_index;
	h_e_index = new int[BLOCK_THREADS*e_prime_gen_grid_size];
	// debug copy stuff to host - check the result of DEIndexPopulateKernel
	CubDebugExit(cudaMemcpy(h_e_index, d_e_index, sizeof(int) * (BLOCK_THREADS*e_prime_gen_grid_size), cudaMemcpyDeviceToHost));
	std::cout<<"The e_index array of size "<<BLOCK_THREADS*e_prime_gen_grid_size<<" is: "<<std::endl;
	for(int i = 0; i < BLOCK_THREADS*e_prime_gen_grid_size; i++){
		std::cout<<h_e_index[i]<<" ";
	}
	std::cout<<std::endl;
#endif

	// find all the combinations - this may later have to be done in consecutive batches
	// 6. Allocate the memory for generating the combinations. TODO: If this memory is out of bounds we may have to batch
	//    the creation.
	//    h_e_prime_scan_end = d_E_prime_size_scan[-1]  is the amount of memory required
	//    d_E_prime
	// 7. Compute the combinations in a kernel with <<<e_prime_gen_grid, BLOCK_THREADS>>> each thread:
	//    reads from e_index[threadNumber]. if it is -1 do nothing and exit.
	//    find the next e_index[threadNumber+...] != -1 with a linear search. Stop search if threadNum+...>=e_index_len
	//    These are index bounds index_low and index_high used to read from the index array
	//    for i of each vertex u in the index array find d_out bounds:
	//       u_low = index[i]
	//       u_high = index[i+1] - make sure i+1 is not out of bounds
	//       u_write_low = d_E_prime_size_scan[i-1] - make sure that this is not out of bounds
	//       u_write_high = d_E_prime_size_scan[i]
	//       scan_write = U_write_low
	//       for i in d_out[u_low:u_high]:
	//          for j in d_out[u_low+1:u_high]
	//             new_edge = edge(u=i,v=j)
	//             assert(scan_write > u_write_high)
	//             d_E_prime[scan_write] = new_edge
	//             scan_write++
	unsigned long *d_E_prime = NULL;
	std::cout<<"Allocating memory of size "<<h_e_prime_scan_end<<" for the d_E_prime"<<std::endl;
	CubDebugExit(cudaMalloc((void**)&d_E_prime, sizeof(unsigned long)*h_e_prime_scan_end));
	std::cout<<"This might fail if its too large. If so you need to batch creation of E_prime. "<<std::endl;
	CubDebugExit(cudaMemset(d_E_prime, 0, sizeof(unsigned long)*h_e_prime_scan_end));

	std::cout<<"Calling kernel to generate EPrime EPrimeComputeKernel..."<<std::endl;
	//EPrimeComputeKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<e_prime_gen_grid_size, BLOCK_THREADS>>>(d_index, d_e_index, d_E_prime_size_scan, h_scan_end+h_cnt_end, d_out, total_num_items, d_E_prime, d_low, d_high);
	EPrimeComputeKernel<unsigned long, BLOCK_THREADS, ITEMS_PER_THREAD><<<e_prime_gen_grid_size, BLOCK_THREADS>>>(d_index, d_e_index, d_E_prime_size_scan, h_scan_end+h_cnt_end, d_out, total_num_items, d_E_prime);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: EPrimeComputeKernel failed due to err code %d.\n", err); return std::vector<edge<unsigned int>>();}


#if 0
	unsigned long *h_E_prime;
	h_E_prime = new unsigned long[h_e_prime_scan_end];
	CubDebugExit(cudaMemcpy(h_E_prime, d_E_prime, sizeof(unsigned long)*h_e_prime_scan_end, cudaMemcpyDeviceToHost));
	std::cout<<"The d_E_prime array of size "<<h_e_prime_scan_end<<" is: "<<std::endl;
	for(unsigned int i = 0; i < h_e_prime_scan_end; i++){
		((edge<unsigned int> *)(&h_E_prime[i]))->print();
		std::cout<<" ";
	}
	std::cout<<std::endl;
#endif

	// ------------------------------------------------------------------------------

	std::cout<<"Copying data to h_in from d_out."<<std::endl;
	// copy data back to host
	//CubDebugExit(cudaMemcpy(h_in, d_out, sizeof(unsigned long) * total_num_items, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_in, d_out, sizeof(unsigned long) * total_num_items, cudaMemcpyDeviceToHost));
	//CubDebugExit(cudaMemcpy(h_in, d_keys.Current(), sizeof(unsigned long) * total_num_items, cudaMemcpyDeviceToHost));
#if 0
	std::cout<<"The modified array: "<<std::endl;
	for(int i = 0; i < total_num_items; i++){
		p[i].print();
		std::cout<<std::endl;
	}

	if (d_cnt) CubDebugExit(cudaFree(d_cnt));
#endif

	std::cout<<"Copying into h_E_prime from d_E_prime."<<std::endl;
	unsigned long *h_E_prime;
	h_E_prime = new unsigned long[h_e_prime_scan_end];
	CubDebugExit(cudaMemcpy(h_E_prime, d_E_prime, sizeof(unsigned long)*h_e_prime_scan_end, cudaMemcpyDeviceToHost));
	std::vector<edge<unsigned int>> E_prime((edge<unsigned int> *)h_E_prime, (edge<unsigned int> *)h_E_prime+h_e_prime_scan_end);

	if (d_in) CubDebugExit(cudaFree(d_in));
	//std::cout<<"Deleting d_out..."<<std::endl;
	//if (d_out) CubDebugExit(cudaFree(d_out));
	if (d_E_prime) CubDebugExit(cudaFree(d_E_prime));
	if (d_key_alt_buf) CubDebugExit(cudaFree(d_key_alt_buf));

	return E_prime;
}

int main(int argc, char* argv[])
{
    std::string fileName="data/A_adj.tsv";
    int c;
    while( ( c = getopt (argc, argv, "f:") ) != -1 )
    {
        switch(c)
        {
            case 'f':
                if(optarg)
                    fileName = optarg;
                break;
        }
    }

    // First read in the data file into memory and perform bucket sort on the first member u of the file
    // Creating an object of CSVWriter
    CSVReader reader(fileName);


    std::cout<<"Size of unsigned int in bits is: "<<sizeof(unsigned int)*8<<" and its max value is: "<<UINT_MAX<<std::endl;
    std::cout<<"Reading in the edges from file"<<reader.fileName<<"....."<<std::endl;
    std::vector<edge<unsigned int> > dataList = reader.getData<unsigned int>();

    std::cout<<"Length of datalist is: "<<dataList.size()<<std::endl;
//    for(int i = 0; i < dataList.size(); i++){
//    	dataList[i].print();
//    	std::cout<<std::endl;
//    }


    std::cout<<"Sorting edges in E...."<<std::endl;
    std::vector<edge<unsigned int>> EPrime = deployKernel<128, 512>((unsigned long *)dataList.data(), (int)dataList.size());

    std::cout<<"Length of EPrime is: "<<EPrime.size()<<std::endl;
    for(int i = 0; i < EPrime.size(); i++){
    	EPrime[i].print();
    	std::cout<<" ";
    }
    std::cout<<std::endl;

    // Last stage - perform triangle count on E(dataList) and EPrime and divide the count found by 3 to get the final count
    // First check if we can simple sort EPrime alone
	unsigned long *d_E_Prime = NULL;
	unsigned long *d_E_Prime_sorted = NULL;
	unsigned int *d_num_elems, *d_offsets, *d_bitnum_beg;
	int h_bitnum_beg[1] = {(sizeof(unsigned long) * 8) - 1};
	unsigned int EPrimeSize = EPrime.size();

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_E_Prime, sizeof(unsigned long) * EPrime.size()));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_E_Prime_sorted, sizeof(unsigned long) * EPrime.size()));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_num_elems, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_offsets, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_bitnum_beg, sizeof(unsigned int)));

	CUDA_CHECK_RETURN(cudaMemcpy(d_E_Prime, EPrime.data(), sizeof(unsigned long) * EPrime.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_num_elems, &EPrimeSize, sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset((void * )d_offsets, 0, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_bitnum_beg, h_bitnum_beg, sizeof(unsigned int), cudaMemcpyHostToDevice));

	radixSort<unsigned long, 128, 1024, 16><<<1, 1>>>(d_E_Prime, d_E_Prime_sorted, d_num_elems, d_offsets, d_bitnum_beg);

//	unsigned long *h_E_Prime_sorted;
//	h_E_Prime_sorted = new unsigned long[EPrimeSize];
//	CubDebugExit(cudaMemcpy(h_E_Prime_sorted, d_E_Prime_sorted, sizeof(unsigned long) * EPrimeSize, cudaMemcpyDeviceToHost));
//	edge<unsigned int> *p;
//	p = (edge<unsigned int> *)h_E_Prime_sorted;
//	std::cout<<"Sorted EPrime is: "<<EPrime.size()<<std::endl;
//	for(int i = 0; i < EPrimeSize; i++){
//		p[i].print();
//		std::cout<<" ";
//	}
//	std::cout<<std::endl;


	return 0;
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
