#include <stdio.h>

//////////////////////////float3////////////////////////////////

inline __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3 &a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ void operator-=(float3 &a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

inline __device__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float3 operator*(float a, float3 b)
{
	return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ float3 operator*(float3 a, int3 b)
{
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

///////////////////////////int3/////////////////////////////////

inline __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ int3 operator-(int3 a, int3 b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ int3 operator+(int3 a, int b)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}

inline __device__ int3 operator-(int3 a, int b)
{
	return make_int3(a.x - b, a.y - b, a.z - b);
}

inline __device__ int3 operator+(int a, int3 b)
{
	return make_int3(a + b.x, a + b.y, a + b.z);
}

inline __device__ int3 operator-(int a, int3 b)
{
	return make_int3(a - b.x, a - b.y, a - b.z);
}

////////////////////////////////////////////////////////////////

inline __device__ int3 clamp(int3 x, int a, int3 b)
{
	return make_int3(max(a, min(x.x, b.x)), max(a, min(x.y, b.y)), max(a, min(x.z, b.z)));
}

inline __device__ int3 clamp(int3 x, int3 a, int b)
{
	return make_int3(max(a.x, min(x.x, b)), max(a.y, min(x.y, b)), max(a.z, min(x.z, b)));
}

inline __device__ int3 floorf(float3 v)
{
	return make_int3(floorf(v.x), floorf(v.y), floorf(v.z));
}

inline __device__ float3 round(float3 v)
{
	return make_float3(round(v.x), round(v.y), round(v.z));
}

inline __device__ int dot(int3 a, int3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ int mod(int a, int b)
{
	int k = a % b;

	return (k < 0) ? (k + b) : k;
}

inline __device__ int3 mod(int3 a, int3 b)
{
	return make_int3(mod(a.x,b.x), mod(a.y,b.y), mod(a.z,b.z));
}

////////////////////////////////////////////////////////////////

struct BinStep
{
	int3 positive, negative;
};

inline __device__ int3 getLocalBinIndex(float3 coordinate, float3 binLength, int3 binDim)
{
	return clamp(floorf(coordinate/binLength), 0, binDim - 1);
}

inline __device__ int getBinIndex(int3 localBinIndex, int3 binDim)
{
	return dot(localBinIndex, make_int3(1, binDim.x, binDim.x*binDim.y));
}

inline __device__ void stepLimit(int &positive, int &negative, int binDim)
{
	if (positive - negative > binDim - 1)
	{
		if (positive > -negative)
		{
			positive = negative + binDim - 1;
		}
		else
		{
			negative = positive - binDim + 1;
		}
	}
}

inline __device__ void lennardJones(float3 &ljF, float &ljU, float3 R, float r2, float eps, float sig)
{
	float sr = sig*sig/r2;
	sr = sr*sr*sr;

	ljF += eps/r2*(2.0f*sr*sr - sr)*R;
	ljU += eps*(sr*sr - sr);
}

__device__ BinStep getBinStep(float3 coordinate, float3 binLength, int3 binDim, int3 localBinIndex, float cutoff)
{
	struct BinStep binStep;

	binStep.positive = clamp(floorf((coordinate + cutoff)/binLength) - localBinIndex, 0, binDim - 1);
	binStep.negative = clamp(floorf((coordinate - cutoff)/binLength) - localBinIndex, 1 - binDim, 0);

	stepLimit(binStep.positive.x, binStep.negative.x, binDim.x);
	stepLimit(binStep.positive.y, binStep.negative.y, binDim.y);
	stepLimit(binStep.positive.z, binStep.negative.z, binDim.z);

	return binStep;
}

////////////////////////////////////////////////////////////////

extern "C"
__global__ void fillBins(float3 *coordinate, int *binIndex, unsigned int *binCount,
		float3 binLength, int3 binDim, unsigned int arraySize)
{
	unsigned int idg = blockIdx.x*blockDim.x + threadIdx.x;

	if (idg < arraySize)
	{
		int idB = getBinIndex(getLocalBinIndex(coordinate[idg], binLength, binDim), binDim);

		binIndex[idg] = idB;
		atomicInc(&binCount[idB], arraySize);
	}
}

extern "C"
__global__ void countingSort(int *binIndex, unsigned int *prefixSum,
		float3 *coordinate, float3 *velocity,
		float3 *coordinateSorted, float3 *velocitySorted,
		unsigned int arraySize)
{
	unsigned int idg = blockIdx.x*blockDim.x + threadIdx.x;

	if (idg < arraySize)
	{
		unsigned int idgSorted = atomicDec(&prefixSum[binIndex[idg]], arraySize) - 1u;

		coordinateSorted[idgSorted] = coordinate[idg];
		velocitySorted[idgSorted] = velocity[idg];
	}
}

extern "C"
__global__ void ljForce(float3 *coordinateSorted, float3 *force,
		float *potentialEnergy, unsigned int *binCount, unsigned int *prefixSum,
		float3 boxSize, float3 binLength, int3 binDim, float cutoff, float eps, float sig,
		unsigned int arraySize)
{
	unsigned int idg = blockIdx.x*blockDim.x + threadIdx.x;

	if (idg < arraySize)
	{
		float3 R;
		float r2;
		int binIndexNeighbour;
		unsigned int ionCount, offset;

		float3 coordinate = coordinateSorted[idg];
		int3 localBinIndex = getLocalBinIndex(coordinate, binLength, binDim);
		struct BinStep binStep = getBinStep(coordinate, binLength, binDim, localBinIndex, cutoff);

		float3 ljF = make_float3(0.0f,0.0f,0.0f);
		float cutoff2 = cutoff*cutoff;
		float ljU = 0.0f;

		for (int dz = binStep.negative.z; dz <= binStep.positive.z; ++dz)
		{
			for (int dy = binStep.negative.y; dy <= binStep.positive.y; ++dy)
			{
				for (int dx = binStep.negative.x; dx <= binStep.positive.x; ++dx)
				{
					binIndexNeighbour = getBinIndex(mod(localBinIndex + make_int3(dx,dy,dz), binDim), binDim);
					ionCount = binCount[binIndexNeighbour];

					if (ionCount == 0u)
					{
						continue;
					}
					offset = prefixSum[binIndexNeighbour];

					for (unsigned int i = offset; i < offset + ionCount; ++i)
					{
						if (i == idg)
						{
							continue;
						}

						R = coordinate - coordinateSorted[i];
						r2 = dot(R, R);

						if (r2 < cutoff2)
						{
							lennardJones(ljF, ljU, R, r2, eps, sig);
							continue;
						}

						R -= boxSize*round(R/boxSize);
						r2 = dot(R, R);

						if (r2 < cutoff2)
						{
							lennardJones(ljF, ljU, R, r2, eps, sig);
						}
					}
				}
			}
		}
		force[idg] = 24.0f*ljF;
		potentialEnergy[idg] = 2.0f*ljU;
	}
}

extern "C"
__global__ void verletPre(float3 *coordinate, float3 *velocity,
		float3 *coordinateSorted, float3 *velocitySorted,
		float3 *force, float3 boxSize, float mass, float dt, unsigned int arraySize)
{
	unsigned int idg = blockIdx.x*blockDim.x + threadIdx.x;

	if (idg < arraySize)
	{
		float3 coord, vel;

		vel = velocitySorted[idg];
		coord = coordinateSorted[idg];

		vel += 0.00482426665f*dt/mass*force[idg]; // 0.00964853329 * 0.5
		coord += dt*vel;

		velocity[idg] = vel;
		coordinate[idg] = coord - boxSize*floorf(coord/boxSize);
	}
}

extern "C"
__global__ void verletPos(float3 *velocitySorted, float3 *force, float *kineticEnergy,
		float mass, float dt, unsigned int arraySize)
{
	unsigned int idg = blockIdx.x*blockDim.x + threadIdx.x;

	if (idg < arraySize)
	{
		float3 vel = velocitySorted[idg];

		vel += 0.00482426665f*dt/mass*force[idg]; // 0.00964853329 * 0.5

		kineticEnergy[idg] = 51.8213479f*mass*dot(vel, vel);
		velocitySorted[idg] = vel;
	}
}



