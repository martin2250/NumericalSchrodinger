const num k = 2;
const num hocl = 2;//powf(8*k, 0.25);
const num xstep = 2*6*hocl/N;
const num stepfactor = hbar * tstep / (xstep * xstep * 2. * mass);

// __device__
cmplx InitialWavefunction(num x, num y)
{
	x = x / hocl;
	y = y / hocl;

	num basex = (hermite(0, x) + hermite(1, x));
	num basey = (hermite(2, y) + hermite(6, y));
	// num basex = (hermite(0, x) + hermite(2, x));
	// num basey = (hermite(2, y) + hermite(1, y));

	cmplx psi;
	psi.re = (basex * basey) * expf(-(x * x + y * y) / 2);
	psi.im = 0;
	return psi;
}

// __device__
num Potential(num x, num y)
{
	return k/2 * x*x + y*y;
}
