#include <Python.h>
#include <math.h>
#include <arrayobject.h>
#define NRANSI
#define TWOPID 6.2831853071795865 /* =2*pi() */


/* See Zechmeister and Kurster 2009 : The generalised Lomb Scargle periodogram + Numerical recipes in C 1992 (for optimization of the code) */



/* fonction permettant de retourner la moyenne et la variance des donnees */
void avevar(double *data, int n, double *ave, double *var)
{
	int j;
	double s,ep;

	for (*ave=0.0,j=0;j<n;j++) *ave += data[j];
	*ave /= n;
	*var=ep=0.0;
	for (j=0;j<n;j++) {
		s=data[j]-(*ave);
		ep += s;
		*var += s*s;
	}
	*var=(*var-ep*ep/n)/(n-1);

}

void period(double *x, double *y, double *weight, int n, double ofac, double hifac, double number_of_night, double *px,
	double *py, double *phi, double *a_cos, double *b_sin, double *c_cte, int np, int *nout, int *jmax, double *prob, double *z0, double *z01)
{
	int i,j;
	double ave,c,effm,expy,pnow,pymax,s,sumc,sumcc,sumcy,sums,sumss,sumcs,
		sumsy,var,xave,xdif,xmax,xmin,sumy,sumyy;
    double yy,yc,ys,cc,ss,cs,d;
	double arg,wtemp,*wi,*wpi,*wpr,*wr;

   wi=(double *) malloc(n*sizeof(double));
	wpi=(double *) malloc(n*sizeof(double));
	wpr=(double *) malloc(n*sizeof(double));
	wr=(double *) malloc(n*sizeof(double));

	avevar(y,n,&ave,&var); /* retourne la moyenne et la variance des donnees */
	xmax=xmin=x[1];

	for (j=0;j<n;j++) { /* scanne les donnees pour avoir l etendu des abscisses */
		if (x[j] > xmax) xmax=x[j];
		if (x[j] < xmin) xmin=x[j];
	}
	xdif=xmax-xmin; /* etendu en abscisse des donnees */
/*    printf("xdif = %f \n",xdif);
    printf("xmax = %f \n",xmax);
    printf("xmin = %f \n",xmin);*/

	xave=0.5*(xmax+xmin); /* moyenne des abscisses */
//~    *nout=(int)(xdif*ofac/(2*delta_T_min));
/*    *nout=np;/*=(int)(xdif*ofac/(2*delta_T_min))*/
    *nout=(int)(xdif*ofac/(2.*hifac)-1);

    /*printf("nout = %d \n", *nout);*/

	/*if (*nout > np) printf("output arrays too short in period"); */
	pymax=0.0;
	pnow=1.0/(xdif*ofac); /* frequence de depart */
  //printf("pnow = %e \n", pnow);
	//pnow=2.0/(xdif); /* frequence de depart */
	for (j=0;j<n;j++) { /* initialise les valeurs pour la recurrence trigonometrique pour chaque point des donnees */
		/*arg=TWOPID*((x[j])*pnow); /* omega=2*pi()*t*f ou t=temps-moyenne temporelle*/
		arg=TWOPID*((x[j]-xmin)*pnow); /* omega=2*pi()*t*f ou t=temps*/
		wpr[j] = -2.0*sin(0.5*arg)*sin(0.5*arg);
		wpi[j]=sin(arg); /* sin(omega*t) */
		wr[j]=cos(arg); /* cos(omega*t) */
		wi[j]=wpi[j];
	}

	for (i=0;i<(*nout);i++) {
		px[i]=pnow;
				//printf("%i over %i ", i,*nout);
        //printf("px[i] = %e %e %e",px[i],pnow,1.0/(ofac*xdif));
        sums=sumc=sumss=sumcc=sumcs=0.0;
		for (j=0;j<n;j++) {
			c=wr[j]; /* c=cos(omega*t) */
			s=wi[j]; /* s=sin(omega*t) */
			sums += weight[j]*s;
			sumc += weight[j]*c;
			sumss += weight[j]*s*s;
			sumcc += weight[j]*c*c;
            sumcs += weight[j]*s*c; /* sin(2x)/2=sin(x)cos(x) */

		}
		sumy=sumyy=sumsy=sumcy=0.0;
		for (j=0;j<n;j++) {
			s=wi[j]; /* s=sin(omega*t) */
			c=wr[j]; /* c=cos(omega*t) */
			yy=y[j]-ave; /*on recentre les mesures autour de la moyenne */
            sumy += weight[j]*yy;
            sumyy += weight[j]*yy*yy;
			sumsy += weight[j]*yy*s; /* (h(j)-moy(h))*sin(omega(t-tau)) */
			sumcy += weight[j]*yy*c; /* (h(j)-moy(h))*cos(omega(t-tau)) */
			wr[j]=((wtemp=wr[j])*wpr[j]-wi[j]*wpi[j])+wr[j];  /* cos(arg+2*pi*nu*t) */
			wi[j]=(wi[j]*wpr[j]+wtemp*wpi[j])+wi[j]; /* sin(arg+2*pi*nu*t) */

      //printf("%f\n",sumy);
		}

      yy = sumyy - sumy*sumy;
      yc = sumcy - sumy*sumc;
      ys = sumsy - sumy*sums;
      cc = sumcc - sumc*sumc;
      ss = sumss - sums*sums;
      cs = sumcs - sumc*sums;

      d = cc*ss - cs*cs;

      a_cos[i]=(yc*ss-ys*cs)/d;
      b_sin[i]=(ys*cc-yc*cs)/d;
      c_cte[i]=sumy - a_cos[i]*sumc - b_sin[i]*sums;
      py[i]=1.0/(yy*d) * (ss*yc*yc + cc*ys*ys - 2.0*cs*ys*yc);
      phi[i]=atan2(a_cos[i],b_sin[i]);

			//printf("test\n");

	  /*if (i==99) printf("freq = %e power = %e phi = %e a**2 = %e b**2 = %e wtau = %e ",pnow,py[i],phi[i],sumcy*sumcy/(sumc),sumsy*sumsy/(sums),wtau);*/
		if (py[i] >= pymax) pymax=py[(*jmax=i)];
		pnow += 1.0/(ofac*xdif); /* frequence suivante */

		//printf("test2\n");


	}
   //printf("var_c = %f ",var);

   /* proprietes statistiques du maximum */
	expy=exp(-pymax); /* exp(-power_max) */
	effm=1.0*(*nout)/(ofac*number_of_night); /* Nfreq/ofac=nombre de freq independantes */
    /*printf("effn = %f ",effm);*/
	*prob=effm*expy; /* P(>z)=1-(1-exp(z))**effm = effm*exp(z)) si significance petite (<0.01 dans ce cas la)*/
   if (*prob > 0.01) *prob=1.0-pow(1.0-expy,effm);/* P(>z)=1-(1-exp(z))**effm ca significence non negligeabe */
	*z0 = -log(1.0-pow(1.0-0.01,1.0/effm));/* z0=-ln(1-(1-p0)**(1/M)) si on trouve un pic au dessus de z0, il a une proba p0 de provenir du bruit*/
    *z01 = -log(1.0-pow(1.0-0.5,1.0/effm));/* z0=-ln(1-(1-p0)**(1/M)) si on trouve un pic au dessus de z0, il a une proba p0 de provenir du bruit*/

	free(wr);
	free(wpr);
	free(wpi);
	free(wi);
}


static PyObject *periodogram(PyObject *self, PyObject *args)
{

	PyArrayObject *a, *b, *c;
	PyArrayObject *aa, *bb, *cc, *dd, *ee, *ff;

	int i, n1, n2, nout, jmax;//, dim[1];
	double *x, *y, *weight, *px, *py, *phi, *a_cos, *b_sin, *c_cte, ofac, hifac, number_of_night, prob, z0, z01, fmax, delta_T_min, ave, var;

	if (!PyArg_ParseTuple(args, "O!O!O!|ddd", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &ofac, &hifac, &number_of_night))
		return NULL;

	x = (double *) (a->data + 0*a->strides[0]);
	y = (double *) (b->data + 0*b->strides[0]);
    weight = (double *) (c->data + 0*c->strides[0]);
	n1 = a->dimensions[0];
   delta_T_min=1e30;
   for (i=0;i<n1-1;i++) { /* scanne les donnees pour avoir l etendu des abscisses */
		if ((x[i+1]-x[i]) < delta_T_min) delta_T_min=x[i+1]-x[i];
	}
    /*delta_T_min=130.0;*/
    /*printf("delta_T_min = %f \n",delta_T_min);*/
	n2 = (int) ((x[n1-1]-x[0])*ofac/(2*hifac)-1);
	px = (double *) malloc(n2*sizeof(double));
	py = (double *) malloc(n2*sizeof(double));
    phi = (double *) malloc(n2*sizeof(double));
    a_cos = (double *) malloc(n2*sizeof(double));
    b_sin = (double *) malloc(n2*sizeof(double));
    c_cte = (double *) malloc(n2*sizeof(double));

	period(x, y, weight, n1, ofac, hifac, number_of_night, px, py, phi, a_cos, b_sin, c_cte, n2, &nout, &jmax, &prob, &z0, &z01);

  avevar(y,n1,&ave,&var);

	//dim[0] = n2;
	// aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
	// bb = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // cc = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // dd = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // ee = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // ff = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
	npy_intp dim[1] = {n2};
	aa = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
	bb = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  cc = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  dd = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  ee = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  ff = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);

	for (i = 0; i < n2; i++) {
		*(double *) (aa->data + i*aa->strides[0]) = px[i];
		*(double *) (bb->data + i*bb->strides[0]) = py[i];
    *(double *) (cc->data + i*cc->strides[0]) = phi[i];
    *(double *) (dd->data + i*dd->strides[0]) = a_cos[i];
    *(double *) (ee->data + i*ee->strides[0]) = b_sin[i];
    *(double *) (ff->data + i*ff->strides[0]) = c_cte[i];
	}

	fmax = px[jmax]; /*=freq_max*/

    free(px);
    free(py);
    free(phi);
    free(a_cos);
    free(b_sin);
    free(c_cte);

/*    return Py_BuildValue("d",var);*/
    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("NNNNNNddddd",aa,bb,cc,dd,ee,ff,fmax,prob,z0,z01,var);
}


static PyMethodDef periodogram_methods[] = {

    {"periodogram", periodogram, METH_VARARGS,
        "Compute the Lomb-Scargle periodogram."},

    {NULL, NULL, 0, NULL}

};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "periodogram_new3",     /* m_name */
    "Very efficient periodogram",                   /* m_doc */
    -1,                    /* m_size */
    periodogram_methods,   /* m_methods */
    NULL,                  /* m_reload */
    NULL,                  /* m_traverse */
    NULL,                  /* m_clear */
    NULL,                  /* m_free */
};
#endif


static PyObject *moduleinit(void)
{
    PyObject *m;

    #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    #else
    m = Py_InitModule3("periodogram_new", periodogram_methods," ");
    #endif

    if (m == NULL) return NULL;

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_periodogram_new3(void)
{
    import_array();
    return moduleinit();
}
#else
PyMODINIT_FUNC initperiodogram_new(void)
{
    (void) moduleinit();

    import_array();
}
#endif


// To test the code
// int main()
// {
// 	int j, n1=100, n2;
// 	double xdiff, hifac, ofac=3., min_P=2.;
// 	int nout, jmax;
// 	double *a, *b, *c, *px, *py, *phi, *a_cos, *b_sin, *c_cte, number_of_night=1., prob, z0, z01;
//
// 	a=(double *) malloc(n1*sizeof(double));
// 	b=(double *) malloc(n1*sizeof(double));
// 	c=(double *) malloc(n1*sizeof(double));
//
// 	for (j=0;j<n1;j++) {
// 		a[j] = j;
// 		b[j] = 2*sin(TWOPID/10.1*j);
// 		c[j] = 0.01;
// 	}
//
// 	xdiff = n1*1.;
// 	hifac = xdiff*ofac / (2 * (1./min_P*xdiff*ofac + 1));
//
// 	n2 = (int) ((a[n1-1]-a[0])*ofac/(2*hifac)-1);
// 	px = (double *) malloc(n2*sizeof(double));
// 	py = (double *) malloc(n2*sizeof(double));
//   phi = (double *) malloc(n2*sizeof(double));
//   a_cos = (double *) malloc(n2*sizeof(double));
//   b_sin = (double *) malloc(n2*sizeof(double));
//   c_cte = (double *) malloc(n2*sizeof(double));
//
// 	printf("Staring periodogram calc.");
// 	period(a, b, c, n1, ofac, hifac, number_of_night, px, py, phi, a_cos, b_sin, c_cte, n2, &nout, &jmax, &prob, &z0, &z01);
// 	printf("Ended periodogram calc.");
//
// 	for (j=0;j<n2;j++) {
// 		printf("%.2f %.2f\n",1./px[j],py[j]);
// 	}
//
// 	return 0;
//
// }


#undef TWOPID
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software 5.){2puDY5m.`. */
