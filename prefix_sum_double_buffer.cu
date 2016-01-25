
__global__ void blockAdd(unsigned int *prefixSum, unsigned int *blockSum,
		unsigned int arraySize)
{
	unsigned int idb = blockIdx.x;
	unsigned int idg = idb*blockDim.x + threadIdx.x;

	if (idg < arraySize && idb > 0u)
	{
		prefixSum[idg] += blockSum[idb - 1u];
	}
}

__global__ void prefixSum(unsigned int *input, unsigned int *output, unsigned int *blockSum,
		unsigned int arraySize)
{
	unsigned int idx = threadIdx.x;
	unsigned int dix = blockDim.x;
	unsigned int idg = blockIdx.x*dix + idx;

	if (idg < arraySize)
	{
		unsigned int in = 1u, out = 0u;
		__shared__ unsigned int buffer[2u][512u];

		buffer[out][idx] = input[idg];
		__syncthreads();

		for (unsigned int offset = 1u; offset < dix; offset *= 2u)
		{
			out = 1u - out;
			in = 1u - out;

			if (idx >= offset)
				buffer[out][idx] = buffer[in][idx] + buffer[in][idx - offset];
			else
				buffer[out][idx] = buffer[in][idx];
			__syncthreads();
		}

		output[idg] = buffer[out][idx];

		if (idx == dix - 1u || idg == arraySize - 1u)
		{
			blockSum[blockIdx.x] = buffer[out][idx];
		}
	}
}
