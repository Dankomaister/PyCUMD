
inline __device__ unsigned int ceilInt(unsigned int x, unsigned int y)
{
	return (x + y - 1u)/y;
}

__global__ void blockAdd(unsigned int *prefixSum, unsigned int *blockSum,
		unsigned int arraySize)
{
	unsigned int idb = blockIdx.x;
	unsigned int dix = blockDim.x;
	unsigned int idg = 2u*idb*dix + threadIdx.x;
	unsigned int idgL, idgH, sum;

	idgL = idg;
	idgH = idg + dix;

	if (idb > 0u)
	{
		sum = blockSum[idb - 1u];

		if (idgL < arraySize)
		{
			prefixSum[idgL] += sum;

			if (idgH < arraySize)
			{
				prefixSum[idgH] += sum;
			}
		}
	}
}

__global__ void prefixSum(unsigned int *input, unsigned int *output, unsigned int *blockSum,
		unsigned int arraySize)
{
	unsigned int idx = threadIdx.x;
	unsigned int dix = blockDim.x;
	unsigned int idg = 2u*blockIdx.x*dix + idx;

	const unsigned int nrMB = 32u;
	unsigned int idxDouble = 2u*idx;
	unsigned int idgL, idgH, idxL, idxH;
	unsigned int blockOffsetL, blockOffsetH;

	__shared__ unsigned int buffer[1024u + 2u*nrMB];

	idgL = idg;
	idgH = idg + dix;

	idxL = idx;
	idxH = idx + dix;

	blockOffsetL = idxL/nrMB;
	blockOffsetH = idxH/nrMB;

	if (idgL < arraySize)
	{
		buffer[idxL + blockOffsetL] = input[idgL];

		if (idgH < arraySize)
		{
			buffer[idxH + blockOffsetH] = input[idgH];
		}
	}
	__syncthreads();

	for (unsigned int offset = 1u; offset < 2u*dix; offset *= 2u)
	{
		idxH = offset*(idxDouble + 2u) - 1u;

		if (idxH < 2u*dix)
		{
			idxL = offset*(idxDouble + 1u) - 1u;

			idxH += idxH/nrMB;
			idxL += idxL/nrMB;

			buffer[idxH] += buffer[idxL];
		}
		__syncthreads();
	}

	for (unsigned int offset = powf(2.0f, ceilf(log2f(ceilInt(dix, 2u)))); offset >= 1u; offset /= 2u)
	{
		idxH = offset*(idxDouble + 3u) - 1u;

		if (idxH < 2u*dix)
		{
			idxL = offset*(idxDouble + 2u) - 1u;

			idxH += idxH/nrMB;
			idxL += idxL/nrMB;

			buffer[idxH] += buffer[idxL];
		}
		__syncthreads();
	}

	idxL = idx;
	idxH = idx + dix;

	if (idgL < arraySize)
	{
		output[idgL] = buffer[idxL + blockOffsetL];

		if (idgH < arraySize)
		{
			output[idgH] = buffer[idxH + blockOffsetH];
		}
	}

	if (idx == dix - 1u && blockIdx.x < gridDim.x - 1u)
	{
		blockSum[blockIdx.x] = buffer[idxH + blockOffsetH];
	}
	else if (idgH == arraySize - 1u)
	{
		blockSum[blockIdx.x] = buffer[idxH + blockOffsetH];
	}
	else if (idgL == arraySize - 1u)
	{
		blockSum[blockIdx.x] = buffer[idxL + blockOffsetL];
	}
}
