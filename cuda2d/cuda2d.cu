#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <math_constants.h>
#include <CImg.h>
#include <boost/math/special_functions/hermite.hpp>
using namespace cimg_library;
using namespace std;
using namespace boost::math;

typedef double num;

struct cmplx
{
	num re;
	num im;
};

const int Nb = 32;
const int N = Nb * 6;
const int windowmultiplier = 3;

const int reducestep = 4;

const int framestep = (int)(5e7/(N*N));

const num hbar = 1.0;
const num mass = 1.0;
const num tstep = 0.8e-5;

#include "harmonicosc.h"


// __global__
// void step(cmplx A[N][N], cmplx B[N][N], num V[N][N])	//unoptimized implementation
// {
// 	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
// 	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
// 	int xt = threadIdx.x;
// 	int yt = threadIdx.y;
//
// 	num dreal = 0.;
// 	num dimag = 0.;
//
// 	dreal += A[(x + 1)%N][y].re + A[(x + N - 1)%N][y].re;
// 	dreal += A[x][(y + 1)%N].re + A[x][(y + N - 1)%N].re;
// 	dreal -= 4 * A[x][y].re;
//
// 	dreal += A[(x + 1)%N][y].im + A[(x + N - 1)%N][y].im;
// 	dreal += A[x][(y + 1)%N].im + A[x][(y + N - 1)%N].im;
// 	dreal -= 4 * A[x][y].im;
//
// 	B[x][y].re = A[x][y].re
// 		- dimag * stepfactor
// 		+ A[x][y].im * V[x][y];
//
// 	B[x][y].im = A[x][y].im
// 		+ dreal * stepfactor
// 		- A[x][y].re * V[x][y];
// }

__global__
void step(cmplx A[N][N], cmplx B[N][N], num V[N][N])
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int xt = threadIdx.x;
	int yt = threadIdx.y;

	__shared__ cmplx As[Nb+2][Nb+2];

	As[xt + 1][yt + 1] = A[x][y];

	if(xt == 0)
		As[0][yt + 1] = A[(N + x - 1)%N][y];
	if(yt == 0)
		As[xt + 1][0] = A[x][(N + y - 1)%N];
	if(xt == (Nb - 1))
		As[Nb + 1][yt + 1] = A[(x + 1)%N][y];
	if(yt == (Nb - 1))
		As[xt + 1][Nb + 1] = A[x][(y + 1)%N];

	__syncthreads();

	num dreal = 0.;
	num dimag = 0.;

	dreal += As[xt + 2][yt + 1].re + As[xt][yt + 1].re;
	dreal += As[xt + 1][yt + 2].re + As[xt + 1][yt].re;
	dreal -= 4 * As[xt + 1][yt + 1].re;

	dimag += As[xt + 2][yt + 1].im + As[xt][yt + 1].im;
	dimag += As[xt + 1][yt + 2].im + As[xt + 1][yt].im;
	dimag -= 4 * As[xt + 1][yt + 1].im;

	B[x][y].re = As[xt + 1][yt + 1].re
		- dimag * stepfactor
		+ As[xt + 1][yt + 1].im * V[x][y];

	B[x][y].im = As[xt + 1][yt + 1].im
		+ dreal * stepfactor
		- As[xt + 1][yt + 1].re * V[x][y];
}

__global__
void getnorm(cmplx A[N][N], num O[N][N])
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	O[x][y] = A[x][y].re * A[x][y].re + A[x][y].im * A[x][y].im;
}

__global__
void getimg(num O[N][N], unsigned char* I, num factor)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

	unsigned char val = (unsigned char)(O[ix][iy] * factor);

	for(int x = 0; x < windowmultiplier; x++)
	{
		for(int y = 0; y < windowmultiplier; y++)
		{
			I[
				(iy * windowmultiplier + y) * N * windowmultiplier
				+ (ix * windowmultiplier + x)
			] = val;
		}
	}
}

__global__
void reduce(num I[N][N], num M[N / reducestep][N / reducestep], num N[N / reducestep][N / reducestep])
{
	int gx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int gy = (blockIdx.y * blockDim.y) + threadIdx.y;

	num maxn = 1e-6;
	num norm = 0;

	for(int x = 0; x < reducestep; x++)
		for(int y = 0; y < reducestep; y++)
		{
			num val = I[gx * reducestep + x][gy * reducestep + y];
			if(val > maxn)
				maxn = val;
			norm += val;
		}

	M[gx][gy] = maxn;
	N[gx][gy] = norm;
}

int main(void)
{
	num hM[N / reducestep][N / reducestep];
	num hN[N / reducestep][N / reducestep];

	cmplx (*dA)[N];
	cmplx (*dB)[N];
	num (*dV)[N];
	num (*dO)[N];	// to hold norm of output

	num (*dM)[N / reducestep];	//to hold max norm
	num (*dN)[N / reducestep];	//to hold sum of norms

	unsigned char *dI;			//hold final image

	cudaMalloc((void**)&dA, N*N*sizeof(cmplx));
	cudaMalloc((void**)&dB, N*N*sizeof(cmplx));
	cudaMalloc((void**)&dV, N*N*sizeof(num));
	cudaMalloc((void**)&dO, N*N*sizeof(num));

	cudaMalloc((void**)&dM, (N / reducestep)*(N / reducestep)*sizeof(num));
	cudaMalloc((void**)&dN, (N / reducestep)*(N / reducestep)*sizeof(num));
	cudaMalloc((void**)&dI, N*windowmultiplier*N*windowmultiplier*sizeof(unsigned char));

	dim3 blocks(N/Nb, N/Nb);
	dim3 threads(Nb, Nb);

	dim3 redblocks(N/(8*reducestep), N/(8*reducestep));
	dim3 redthreads(8, 8);

	cmplx hInitA[N][N];
	num hInitV[N][N];
	for(int x = 0; x < N; x++)
	for(int y = 0; y < N; y++)
	{
		num xp = ((num)(x - (N/2)) + 0.5) * xstep;
		num yp = ((num)(y - (N/2)) + 0.5) * xstep;

		hInitA[x][y] = InitialWavefunction(xp, yp);
		hInitV[x][y] = Potential(xp, yp) * tstep / hbar;
	}

	cudaMemcpy(dA, hInitA, N*N*sizeof(cmplx), cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hInitV, N*N*sizeof(num), cudaMemcpyHostToDevice);


	CImg<unsigned char> canvas( N*windowmultiplier, N*windowmultiplier, 1, 1, 0);
	CImgDisplay window( canvas, "Schrodinger 2D");

	int frame = 0;

	while (!window.is_closed())
	{
		auto startframe = std::chrono::high_resolution_clock::now();

		getnorm<<<blocks, threads>>>(dA, dO);
		reduce<<<redblocks, redthreads>>>(dO, dM, dN);

		cudaDeviceSynchronize();
		cudaMemcpy(hM, dM, (N / reducestep)*(N / reducestep)*sizeof(num), cudaMemcpyDeviceToHost);
		cudaMemcpy(hN, dN, (N / reducestep)*(N / reducestep)*sizeof(num), cudaMemcpyDeviceToHost);

		num fac = 1e-6;
		num norm = 0;

		for(int x = 0; x < (N / reducestep); x++)
			for(int y = 0; y < (N / reducestep); y++)
			{
				if(hM[x][y] > fac)
					fac = hM[x][y];
				norm += hN[x][y];
			}

		fac = 255. / fac;

		getimg<<<blocks, threads>>>(dO, dI, fac);

		cudaDeviceSynchronize();

		for(int j = 0; j < framestep; j+=2)
		{
			step<<<blocks, threads>>>(dA, dB, dV);
			step<<<blocks, threads>>>(dB, dA, dV);
		}

		cudaMemcpy(canvas.data(), dI, N*windowmultiplier*N*windowmultiplier*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		auto startblock = std::chrono::high_resolution_clock::now();

		window.display(canvas);
		auto endblock = std::chrono::high_resolution_clock::now();

		auto endframe = std::chrono::high_resolution_clock::now();
		auto framems = std::chrono::duration_cast<std::chrono::microseconds>(endframe - startframe).count() / 1000;
		auto blockms = std::chrono::duration_cast<std::chrono::microseconds>(endblock - startblock).count() / 1000;

		cout << "\rframe " << setw(5) << frame++ << " "
			<< setw(4) << framems << "ms "
			<< setw(3) << 1000 / framems << "fps | "
			//<< setw(4) << blockms << "ms (block)| "
			<< "norm: " << norm << endl;
	}

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dO);
	cudaFree(dV);
	cudaFree(dM);
	cudaFree(dN);
	cudaFree(dI);

	return 0;
}
