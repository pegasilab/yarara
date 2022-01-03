#include <Python.h>
#include <math.h>
#include <arrayobject.h>
#include <stdio.h>
#include <gsl/gsl_multifit.h>
#define NRANSI
#define CLUM 299792458. // vitesse de la lumiere en m/s

double rndup(double n,int nb_decimal)//round up a double type at nb_decimal
{
    double t;
    t=n*pow(10,nb_decimal) - floor(n*pow(10,nb_decimal));
    if (t>=0.5)
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = ceil(n);
        n/=pow(10,nb_decimal);
    }
    else
    {
        n*=pow(10,nb_decimal);//where n is the multi-decimal double
        n = floor(n);
        n/=pow(10,nb_decimal);
    }
    return n;
}

void avevar(double *data, int n, double *ave)
{
	int j;

	for (*ave=0.0,j=0;j<n;j++) *ave += data[j];
	*ave /= n;
}

int compare (const void * a, const void * b)
{
    const double *da = (const double *) a;
    const double *db = (const double *) b;

    return (*da > *db) - (*da < *db);
}

void median(double *data, int n1, double *med)
{
    int mean_len;

    qsort(data, n1, sizeof(double), compare);
    mean_len = rndup(n1/2.,0)-5;
    *med = data[mean_len];
}


//Calcul le decalage (ou l elargissement) d une raie en tenant compte de la vitesse et de la longueur d onde, en angstrom
// line_width est la largeur de la raie en vitesse
double Delta_lambda(double line_width, double lambda_line0)
{
    double c=299792458.;
    double line_spread=0.;
	double beta;

	// cas relativiste et symmetrique voir "functions.py"
	beta = line_width/c;
	line_spread = -1 * lambda_line0 * (1 - sqrt((1+beta)/(1-beta)));

	// v/c*wavelength, cas non relaticviste et non symetrique
    //line_spread = line_width/c*lambda_line0;

    return line_spread;
}


void obtain_same_sampling(double *freq, int n1,double *freq_oversampled, int n2, double *flux_oversampled,double *freq_same, double *flux_same, int *index_same, double *diff)
{
    int i,j=0;
    double diff_min;

    for (i=0;i<n1;i++)
    {
        //if (i%1000==0) printf("%d over %d\n",i,n1);
        diff_min=1.e30;
        while((fabs(freq_oversampled[j]-freq[i]) < diff_min) && (j < n2))
        {
            diff_min = fabs(freq_oversampled[j]-freq[i]);
            j += 1;
        }
		j -= 1;
        freq_same[i] = freq_oversampled[j];
        flux_same[i] = flux_oversampled[j];
        index_same[i] = j;
		diff[i] = freq[i]-freq_oversampled[j];
    }
}

void calculate_RV_with_BOUCHY(double *wavelength,double *spectrum_raw,double *spectrum,double *spectrum_master, int n1, double *RV, double *sig_RV, double detector_noise)
{
    /*
    See function "calculate_RV_with_BOUCHY(wavelength,spectrum,spectrum_master)" in "compare_e2ds_for_line_all_orders_RV_with_master_dace_no_corr.py"
    Methode BOUCHY, see Bouchy et al. 2001, ATTENTION formule 9 fausse car W est toujours positif car au carre, donc abs(sqrt(W/sig_A[:-1]**2)) = abs(1./(wavelength[:-1]*derivative)*W), mais si on onleve les valeurs absolue, c est faux....
    Delta_A = dI/dl.Delta_l, where l is wavelength and I is intensity, dI/dl is the derivative
    sig_Delta_A = sqrt(sqrt(spectrume) + detector_noise**2)
    Delta_l = Delta_A / (dI/dl)
    sig(Delta_l) = sqrt(1/(dI/dl)^2*sig(Delta_A)^2 + Order(sig(di/dl)*2)), sig(di/dl) negligeable because measured on master at very high S/N
     -> sig(Delta_l) = sig(Delta_A)/(dI/dl)

    sum((A-A0)/(lambda*dA0/dlambda)*(lambda*dA0/dlambda)**2/(A+sig_detector**2)) / sum((lambda*dA0/dlambda)**2/(A+sig_detector**2))

    On utilise spectrum_raw pour determiner l erreur, car le vrai signal est signal avant la correction du BLAZE, de l extinction atmospherique
    et de la distribution de l energie spectrumale (Planck function)
    */

    int i;
    double upper_term=0,lower_term=0;
    double Delta_A, sig_A, derivative, W;

    for (i=1;i<n1;i++)
    {
        Delta_A = (spectrum[i-1] - spectrum_master[i-1]);
        sig_A = sqrt(spectrum_raw[i-1] + detector_noise*detector_noise); // =A_rms in paper. photon noise = sqrt(flux). The flux is in electron ~ photons, we normalize here
        derivative = (spectrum_master[i] - spectrum_master[i-1]) / (wavelength[i] - wavelength[i-1]); // Noise on derivative negligible because very high S/N
        W = pow(wavelength[i-1]*derivative,2)/pow(sig_A,2);
        // RV = sum(Delta_A[:-1]/(wavelength[:-1]*derivative)*W)/sum(W)*CLUM
        upper_term += Delta_A/(wavelength[i-1]*derivative)*W;
        lower_term += W;
    }
    *RV = upper_term / lower_term * CLUM;
    *sig_RV = 1./pow(lower_term,1/2.) * CLUM;
}


void get_RV_line_by_line_c(double *vrad_ccf, double *RV_vect, double *sig_RV_vect, double *chisq_vect, int n1, double *wavelength, double *spectrum_raw, double *spectrum, int n2, double *wavelength_master, double *spectrum_master, int n3, double detector_noise, double **fit_para) //n1 = len(vrad_ccf), n2 = len(avelength), n3 = len(wavelength_master)
{
    int i,j, *index_same;
    double max_spectrum_master_same=0,sig_spectrum, chisq=-10;
    double *wavelength_master_same, *spectrum_master_same, *diff, *spectrum_master_norm, *wavelength_shifted_to_0, *wavelength_master_shift;

    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;

    // Allocating memory
    wavelength_master_same  = (double *) malloc(n2*sizeof(double));
    spectrum_master_same    = (double *) malloc(n2*sizeof(double));
    index_same              = (int *) malloc(n2*sizeof(int));
    diff                    = (double *) malloc(n2*sizeof(double));
    spectrum_master_norm    = (double *) malloc(n2*sizeof(double));
    wavelength_shifted_to_0 = (double *) malloc(n2*sizeof(double));
    wavelength_master_shift = (double *) malloc(n3*sizeof(double));

    // Allocating memory for GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    y = gsl_vector_alloc (n2);
    w = gsl_vector_alloc (n2);
//    X = gsl_matrix_alloc (n2, 3);
//    c = gsl_vector_alloc (3);
//    cov = gsl_matrix_alloc (3, 3);
    X = gsl_matrix_alloc (n2, 2);
    c = gsl_vector_alloc (2);
    cov = gsl_matrix_alloc (2, 2);

    //calculate the shift of the spectrumal line in the master, and then get the same sampling as the spectrumal line in the spectrum. Remember that the spectrumal line in the master is oversampled by fitting a high order polynomial.
    for (i=0;i<n1;i++)
    {

        double RV=100000, sig_RV=100000,sum_spectrum_master_norm=0,min_spectrum_master_norm=100000;

        for (j=0;j<n3;j++) wavelength_master_shift[j] = wavelength_master[j] + Delta_lambda(vrad_ccf[i],wavelength_master[j]); // calculate the position of the spectrumal lines as a function of the spectrumal line in the master (given by ccf_vrad)
        obtain_same_sampling(wavelength,n2,wavelength_master_shift,n3,spectrum_master,wavelength_master_same,spectrum_master_same,index_same,diff); //wavelength_master_same,spectrum_master_same have the same sampling as wavelength,spectrum, to be able to calculate the Bouchy drift after.

        for (j=0;j<n2;j++)
        {
            if (spectrum_master_same[j] > max_spectrum_master_same) max_spectrum_master_same = spectrum_master_same[j];
        }

        // calculate the RV for the given shift of the spectrumal line in the master (given by ccf_vrad[i])

        //adjust the master to the level of the spectrum, to have the correct flux and therefore errors and then fit with a linlsq the scaling factor, the level and slope of the spectrum to match the master.

        for (j=0;j<n2;j++)
        {
//            norm_factor = 1./max_spectrum_master_same*max_spectrum;
//            //printf("%e  ",norm_factor);
//            spectrum_master_norm[j] = spectrum_master_same[j] * norm_factor;
//            //wavelength_shifted_to_0[j] = wavelength[j] - wavelength[0];
//            sig_spectrum_master_norm = pow(spectrum_master_same[j],1/2.); //We do not had "+ detector_noise**2" because we make the assumption that the master has no noise. Here we use the normalized master to the spectrum we are looking, therefore if the spectrum we are looking have a extremely low SNR, adding the detector noise to the master can have a huge effect.
//            //printf("%.4f\n",sig_spectrum_master_norm);
//
//            // Initializing the variables for the GSL linear lsq
//            gsl_matrix_set (X, j, 0, spectrum[j]);
//            //gsl_matrix_set (X, j, 1, wavelength_shifted_to_0[j]);
//            //gsl_matrix_set (X, j, 2, 1.0);
//            gsl_matrix_set (X, j, 1, 1.0);
//            gsl_vector_set (y, j, spectrum_master_norm[j]);
//            gsl_vector_set (w, j, 1.0/(sig_spectrum_master_norm*sig_spectrum_master_norm));

            sig_spectrum = pow(spectrum_raw[j] + pow(detector_noise,2),1/2.);
            //printf("%.4f\n",sig_spectrum_master_norm);

            // Initializing the variables for the GSL linear lsq
            gsl_matrix_set (X, j, 0, spectrum_master_same[j]);
            //gsl_matrix_set (X, j, 1, wavelength_shifted_to_0[j]);
            //gsl_matrix_set (X, j, 2, 1.0);
            gsl_matrix_set (X, j, 1, 1.0);
            gsl_vector_set (y, j, spectrum[j]);
            gsl_vector_set (w, j, 1.0/(sig_spectrum*sig_spectrum));
        }

        // Fitting linlsq with GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
        {
            //gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n2, 3);
            gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n2, 2);
            gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
            gsl_multifit_linear_free (work);
        }
        //correct the spectrum with the best fitted value from the linlsq
        //for (j=0;j<n2;j++) spectrum_master_norm[j] = gsl_vector_get(c,(0)) * spectrum_master_same[j] + gsl_vector_get(c,(1)) * wavelength_shifted_to_0[j] + gsl_vector_get(c,(2));
        for (j=0;j<n2;j++) spectrum_master_norm[j] = gsl_vector_get(c,(0)) * spectrum_master_same[j] + gsl_vector_get(c,(1));

        //save the paramaters into the vector "fit_para"
        fit_para[i][0] = gsl_vector_get(c,(0));
        fit_para[i][1] = gsl_vector_get(c,(1));

        //return a bad flag if sum(spectrum_selected_for_line_corrected) == 0 or sort(spectrum_selected_for_line_corrected)[0] < 0
        for (j=0;j<n2;j++)
        {
            sum_spectrum_master_norm += spectrum_master_norm[j];
            if (spectrum_master_norm[j] < min_spectrum_master_norm) min_spectrum_master_norm = spectrum_master_norm[j];
        }
        if (sum_spectrum_master_norm == 0.0)// || spectrum_master_norm[0] <= 0, we removed this part because if the spectrum is always decreasing or increasing, there is a possibility that the first or last value is smaller than 0
        {
            RV_vect[i] = 100000;
            sig_RV_vect[i] = 100000;
            chisq_vect[i] = 100000;
        }
        else
        {
            // calculate the BOUVHY RV
            calculate_RV_with_BOUCHY(wavelength,spectrum_raw,spectrum,spectrum_master_norm,n2,&RV,&sig_RV,detector_noise);
            RV_vect[i] = RV;
            sig_RV_vect[i] = sig_RV;
            chisq_vect[i] = chisq/(1.*(n2-3.0));
        }
    }

    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);

    free(wavelength_master_same);
    free(spectrum_master_same);
    free(index_same);
    free(diff);
    free(spectrum_master_norm);
    free(wavelength_shifted_to_0);
    free(wavelength_master_shift);
}


void get_RV_line_by_line_Bouchy_without_offset(double *normalization, double *RV, double *sig_RV, double *constant, double *chi2, double freq_line, double *wavelength, double *spectrum_raw, double *spectrum, int n2, double *wavelength_master, double *spectrum_master, int n3, double detector_noise,double vrad_ccf) //n1 = len(vrad_ccf), n2 = len(avelength), n3 = len(wavelength_master)
{
    int j, nb_para=2, *index_same; // use n2-1, because this is the number of data points (because of the derivative we loose one)
    double sig_spectrum, chisq=-10, delta_lambda=1e30, sig_delta_lambda=1e30;
    double *wavelength_master_same, *spectrum_master_same, *diff, *derivative_spectrum_master, *derivative_spectrum_master_same, *wavelength_master_shift;
    double x0,x1,y0,y1,y0_deriv,y1_deriv,slope,slope_deriv,ordinate_cross,ordinate_cross_deriv;


    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;

    // Allocating memory
    derivative_spectrum_master      = (double *) malloc((n3-1)*sizeof(double)); //one less data point for the derivative
    wavelength_master_same          = (double *) malloc(n2*sizeof(double));
    index_same                      = (int *) malloc(n2*sizeof(int));
    diff                            = (double *) malloc(n2*sizeof(double));
    spectrum_master_same            = (double *) malloc(n2*sizeof(double));
    derivative_spectrum_master_same = (double *) malloc(n2*sizeof(double));
    wavelength_master_shift         = (double *) malloc(n3*sizeof(double));

    //Initialisation
    for (j=0;j<n3-1;j++) derivative_spectrum_master[j]=0.0;
    for (j=0;j<n2;j++) wavelength_master_same[j]=0.0;
    for (j=0;j<n2;j++) index_same[j]=0.0;
    for (j=0;j<n2;j++) diff[j]=0.0;
    for (j=0;j<n2;j++) spectrum_master_same[j]=0.0;
    for (j=0;j<n2;j++) derivative_spectrum_master_same[j]=0.0;
    for (j=0;j<n3;j++) wavelength_master_shift[j]=0.0;

    // Allocating memory for GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    y = gsl_vector_alloc (n2);
    w = gsl_vector_alloc (n2);
    X = gsl_matrix_alloc (n2, nb_para);
    c = gsl_vector_alloc (nb_para);
    cov = gsl_matrix_alloc (nb_para, nb_para);

    //vrad_ccf should be equal to "-berv*1000.+vrad_diff_spectr_ref" to center the spectral line of the master on the spectral line of the stpectrum
    for (j=0;j<n3;j++)
    {
        wavelength_master_shift[j] = wavelength_master[j] + Delta_lambda(vrad_ccf,wavelength_master[j]); // calculate the position of the spectrum line as a function of the spectral line in the master (given by ccf_vrad)
    }

    //calculate_derivatives
    for (j=0;j<n3-1;j++)//n3-1 because the derivative can be calculated only for one point less
    {
        derivative_spectrum_master[j] = (spectrum_master[j+1] - spectrum_master[j]) / (wavelength_master_shift[j+1] - wavelength_master_shift[j]);
    }

    // Get the closest points in the master that correspond to the spectrum, and then in the next step linearly interpolate from there
    obtain_same_sampling(wavelength,n2,wavelength_master_shift,n3,spectrum_master,wavelength_master_same,spectrum_master_same,index_same,diff); //wavelength_master_same,spectrum_master_same have the same sampling as wavelength,spectrum, to be able to calculate the Bouchy drift after.

    //linear_interpolation
    for (j=0;j<n2;j++)
    {
        if (diff[j] < 0)  //diff[i] = freq[i]-freq_oversampled[j] as defined in the function "obtain_same_sampling". diff < 0 => wavelength < wavelength_master_shift
        {
//            printf("diff < 0\t");
//            printf("%.8f\t%.8f\t%.8f\t%.8f\n",wavelength_master_shift[index_same[j]-1],wavelength[j],wavelength_master_shift[index_same[j]],diff[j]);
            x0 = wavelength_master_shift[index_same[j]-1];
            x1 = wavelength_master_shift[index_same[j]];
            y0 = spectrum_master[index_same[j]-1];
            y1 = spectrum_master[index_same[j]];
            y0_deriv = derivative_spectrum_master[index_same[j]-1];
            y1_deriv = derivative_spectrum_master[index_same[j]];
        }
        else //diff >= 0 => wavelength >= wavelength_master_shift
        {
//            printf("diff >= 0\t");
//            printf("%.8f\t%.8f\t%.8f\t%.8f\n",wavelength_master_shift[index_same[j]],wavelength[j],wavelength_master_shift[index_same[j]+1],diff[j]);
            x0 = wavelength_master_shift[index_same[j]];
            x1 = wavelength_master_shift[index_same[j]+1];
            y0 = spectrum_master[index_same[j]];
            y1 = spectrum_master[index_same[j]+1];
            y0_deriv = derivative_spectrum_master[index_same[j]];
            y1_deriv = derivative_spectrum_master[index_same[j]+1];
        }
        slope = (y1-y0)/(x1-x0);
        ordinate_cross = y1 - slope*x1;
        slope_deriv = (y1_deriv-y0_deriv)/(x1-x0);
        ordinate_cross_deriv = y1_deriv - slope_deriv*x1;

        spectrum_master_same[j]             = slope*wavelength[j] + ordinate_cross;
        derivative_spectrum_master_same[j]  = slope_deriv*wavelength[j] + ordinate_cross_deriv;
        //printf("diff = %.4f\tw=%.4f\tw_master0=%.4f\tw_master1=%.4f\ty0=%.4f\ty1=%.4f\ty_interpol=%.4f\n",diff[j],wavelength[j],x0,x1,y0,y1,spectrum_master_same[j]);
    }

    //BOUCHY method using a linear least square.
    for (j=0;j<n2;j++)
    {
        sig_spectrum = pow(spectrum_raw[j] + pow(detector_noise,2),1/2.); //We assume that the noise on the master is negligible, therefore no noise on the derivative

        // Initializing the variables for the GSL linear lsq
        gsl_matrix_set (X, j, 0, spectrum_master_same[j]);
        gsl_matrix_set (X, j, 1, derivative_spectrum_master_same[j]);
        //gsl_matrix_set (X, j, 2, 1.0);
        gsl_vector_set (y, j, spectrum[j]);
        gsl_vector_set (w, j, 1.0/(sig_spectrum*sig_spectrum));
    }

    // Fitting linlsq with GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n2, nb_para);
    gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
    gsl_multifit_linear_free (work);


    //gsl_vector_get(c,(0)) = c
    //gsl_vector_get(c,(1)) = c . Dlambda
    //Dlambda = (c . Dlambda) / c
    //sig_Dlambda = sqrt((1/c)^2 * sig_cDlambda^2 + (-cDlambda/c^2)^2 * sig_c^2)
    //RV = -1. * Dlambda/lambda*c_lum + vrad_ccf #The -1 comes from testing. vrad_ccf because with shifted the entire spectrum by this value, so we have to add it
    //sig_RV = sig_Dlambda/lambda*c_lum
    // BE CAREFULL gsl_matrix_get(cov,1,1) = sigma_(c . Dlambda)**2 and gsl_matrix_get(cov,0,0) = sigma_c**2. Already squared, so BE CAREFULL WHEN CALCULATING THE ERROR

    *normalization = gsl_vector_get(c,(0));
    delta_lambda = gsl_vector_get(c,(1)) / gsl_vector_get(c,(0));
    //gsl_matrix_get(cov,1,1) equal to sigma**2, so no need to put the power 2
    sig_delta_lambda = pow(pow(1./gsl_vector_get(c,(0)),2)*gsl_matrix_get(cov,1,1) + pow(-gsl_vector_get(c,(1))/pow(gsl_vector_get(c,(0)),2),2)*gsl_matrix_get(cov,0,0),1/2.);
    *RV = -1. * delta_lambda / freq_line * CLUM + vrad_ccf;
    *sig_RV = sig_delta_lambda / freq_line * CLUM;
    //*constant = gsl_vector_get(c,(2));
    *chi2 = chisq / (n2 - nb_para - 1);


    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);

    free(wavelength_master_same);
    free(derivative_spectrum_master);
    free(spectrum_master_same);
    free(derivative_spectrum_master_same);
    free(index_same);
    free(diff);
    free(wavelength_master_shift);
}

void get_RV_line_by_line_Bouchy_with_offset(double *normalization, double *RV, double *sig_RV, double *constant, double *chi2, double freq_line, double *wavelength, double *spectrum_raw, double *spectrum, int n2, double *wavelength_master, double *spectrum_master, int n3, double detector_noise,double vrad_ccf) //n1 = len(vrad_ccf), n2 = len(avelength), n3 = len(wavelength_master)
{
    int j, nb_para=3, *index_same; // use n2-1, because this is the number of data points (because of the derivative we loose one)
    double sig_spectrum, chisq=-10, delta_lambda=1e30, sig_delta_lambda=1e30;
    double *wavelength_master_same, *spectrum_master_same, *diff, *derivative_spectrum_master, *derivative_spectrum_master_same, *wavelength_master_shift;
    double x0,x1,y0,y1,y0_deriv,y1_deriv,slope,slope_deriv,ordinate_cross,ordinate_cross_deriv;


    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;

    // Allocating memory
    derivative_spectrum_master      = (double *) malloc((n3-1)*sizeof(double)); //one less data point for the derivative
    wavelength_master_same          = (double *) malloc(n2*sizeof(double));
    index_same                      = (int *) malloc(n2*sizeof(int));
    diff                            = (double *) malloc(n2*sizeof(double));
    spectrum_master_same            = (double *) malloc(n2*sizeof(double));
    derivative_spectrum_master_same = (double *) malloc(n2*sizeof(double));
    wavelength_master_shift         = (double *) malloc(n3*sizeof(double));

    // Allocating memory for GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    y = gsl_vector_alloc (n2);
    w = gsl_vector_alloc (n2);
    X = gsl_matrix_alloc (n2, nb_para);
    c = gsl_vector_alloc (nb_para);
    cov = gsl_matrix_alloc (nb_para, nb_para);

    //vrad_ccf should be equal to "-berv*1000.+vrad_diff_spectr_ref" to center the spectral line of the master on the spectral line of the stpectrum
    for (j=0;j<n3;j++)
    {
        wavelength_master_shift[j] = wavelength_master[j] + Delta_lambda(vrad_ccf,wavelength_master[j]); // calculate the position of the spectrum line as a function of the spectral line in the master (given by ccf_vrad)
    }

    //calculate_derivatives
    for (j=0;j<n3-1;j++)//n3-1 because the derivative can be calculated only for one point less
    {
        derivative_spectrum_master[j] = (spectrum_master[j+1] - spectrum_master[j]) / (wavelength_master_shift[j+1] - wavelength_master_shift[j]);
    }

    //obtain_same_sampling
    obtain_same_sampling(wavelength,n2,wavelength_master_shift,n3,spectrum_master,wavelength_master_same,spectrum_master_same,index_same,diff); //wavelength_master_same,spectrum_master_same have the same sampling as wavelength,spectrum, to be able to calculate the Bouchy drift after.

    //linear_interpolation
    for (j=0;j<n2;j++)
    {
        if (diff[j] < 0)  //diff[i] = freq[i]-freq_oversampled[j] as defined in the function "obtain_same_sampling". diff < 0 => wavelength < wavelength_master_shift
        {
            x0 = wavelength_master_shift[index_same[j]-1];
            x1 = wavelength_master_shift[index_same[j]];
            y0 = spectrum_master[index_same[j]-1];
            y1 = spectrum_master[index_same[j]];
            y0_deriv = derivative_spectrum_master[index_same[j]-1];
            y1_deriv = derivative_spectrum_master[index_same[j]];
        }
        else
        {
            x0 = wavelength_master_shift[index_same[j]];
            x1 = wavelength_master_shift[index_same[j]+1];
            y0 = spectrum_master[index_same[j]];
            y1 = spectrum_master[index_same[j]+1];
            y0_deriv = derivative_spectrum_master[index_same[j]];
            y1_deriv = derivative_spectrum_master[index_same[j]+1];
        }
        slope = (y1-y0)/(x1-x0);
        ordinate_cross = y1 - slope*x1;
        slope_deriv = (y1_deriv-y0_deriv)/(x1-x0);
        ordinate_cross_deriv = y1_deriv - slope_deriv*x1;

        spectrum_master_same[j]             = slope*wavelength[j] + ordinate_cross;
        derivative_spectrum_master_same[j]  = slope_deriv*wavelength[j] + ordinate_cross_deriv;
        //printf("diff = %.4f\tw=%.4f\tw_master0=%.4f\tw_master1=%.4f\ty0=%.4f\ty1=%.4f\ty_interpol=%.4f\n",diff[j],wavelength[j],x0,x1,y0,y1,spectrum_master_same[j]);
    }

    //BOUCHY method using a linear least square.
    for (j=0;j<n2;j++)
    {
        sig_spectrum = pow(spectrum_raw[j] + pow(detector_noise,2),1/2.); //We assume that the noise on the master is negligible, therefore no noise on the derivative

        // Initializing the variables for the GSL linear lsq
        gsl_matrix_set (X, j, 0, spectrum_master_same[j]);
        gsl_matrix_set (X, j, 1, derivative_spectrum_master_same[j]);
        gsl_matrix_set (X, j, 2, 1.0);
        gsl_vector_set (y, j, spectrum[j]);
        gsl_vector_set (w, j, 1.0/(sig_spectrum*sig_spectrum));
    }

    // Fitting linlsq with GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n2, nb_para);
    gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
    gsl_multifit_linear_free (work);


    //gsl_vector_get(c,(0)) = c
    //gsl_vector_get(c,(1)) = c . Dlambda
    //Dlambda = (c . Dlambda) / c
    //sig_Dlambda = sqrt((1/c)^2 * sig_cDlambda^2 + (-cDlambda/c^2)^2 * sig_c^2)
    //RV = -1. * Dlambda/lambda*c_lum + vrad_ccf #The -1 comes from testing. vrad_ccf because with shifted the entire spectrum by this value, so we have to add it
    //sig_RV = sig_Dlambda/lambda*c_lum
    // BE CAREFULL gsl_matrix_get(cov,1,1) = sigma_(c . Dlambda)**2 and gsl_matrix_get(cov,0,0) = sigma_c**2. Already squared, so BE CAREFULL WHEN CALCULATING THE ERROR

    *normalization = gsl_vector_get(c,(0));
    delta_lambda = gsl_vector_get(c,(1)) / gsl_vector_get(c,(0));
    //gsl_matrix_get(cov,1,1) equal to sigma**2, so no need to put the power 2
    sig_delta_lambda = pow(pow(1./gsl_vector_get(c,(0)),2)*gsl_matrix_get(cov,1,1) + pow(-gsl_vector_get(c,(1))/pow(gsl_vector_get(c,(0)),2),2)*gsl_matrix_get(cov,0,0),1/2.);
    *RV = -1. * delta_lambda / freq_line * CLUM + vrad_ccf;
    *sig_RV = sig_delta_lambda / freq_line * CLUM;
    *constant = gsl_vector_get(c,(2));
    //printf("%.2f",gsl_vector_get(c,(2)));
    *chi2 = chisq / (n2 - nb_para - 1);


    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);

    free(wavelength_master_same);
    free(derivative_spectrum_master);
    free(spectrum_master_same);
    free(derivative_spectrum_master_same);
    free(index_same);
    free(diff);
    free(wavelength_master_shift);
}




void get_RV_line_by_line_Bouchy_with_offset_second_order_deriv(double *para0, double *para1, double *para2, double *RV, double *sig_RV, double *constant, double *chi2, double freq_line, double *wavelength, double *spectrum_raw, double *spectrum, int n2, double *wavelength_master, double *spectrum_master, int n3, double detector_noise,double vrad_ccf,double *wavelength_master_shift,double *derivative_spectrum_master,double *derivative2_spectrum_master,double *spectrum_master_same,double *derivative_spectrum_master_same,double *derivative2_spectrum_master_same) //n1 = len(vrad_ccf), n2 = len(avelength), n3 = len(wavelength_master)
{
    int j, nb_para=3, *index_same; // use n2-2, because this is the number of data points (because of the 2nd derivative we loose two)
    double sig_spectrum, chisq=-10, delta_lambda=1e30, sig_delta_lambda=1e30;
    double *wavelength_master_same, *diff;
    //double *wavelength_master_shift, *derivative_spectrum_master, *derivative2_spectrum_master;
    //double *spectrum_master_same, *derivative_spectrum_master_same, *derivative2_spectrum_master_same;
    double x0,x1,y0,y1,y0_deriv,y1_deriv,y0_deriv2,y1_deriv2,slope,slope_deriv,slope_deriv2,ordinate_cross,ordinate_cross_deriv,ordinate_cross_deriv2;

    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;

    // Allocating memory
//    wavelength_master_shift          = (double *) malloc(n3*sizeof(double));
//    derivative_spectrum_master       = (double *) malloc((n3-1)*sizeof(double)); //one less data point for the derivative
//    derivative2_spectrum_master      = (double *) malloc((n3-2)*sizeof(double)); //two less data point for the 2nd derivative
    index_same                       = (int *) malloc(n2*sizeof(int));
    diff                             = (double *) malloc(n2*sizeof(double));
    wavelength_master_same           = (double *) malloc(n2*sizeof(double));
//    spectrum_master_same             = (double *) malloc(n2*sizeof(double));
//    derivative_spectrum_master_same  = (double *) malloc(n2*sizeof(double));
//    derivative2_spectrum_master_same = (double *) malloc(n2*sizeof(double));

    // Allocating memory for GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    y = gsl_vector_alloc (n2);
    w = gsl_vector_alloc (n2);
    X = gsl_matrix_alloc (n2, nb_para);
    c = gsl_vector_alloc (nb_para);
    cov = gsl_matrix_alloc (nb_para, nb_para);

    //vrad_ccf should be equal to "-berv*1000.+vrad_diff_spectr_ref" to center the spectral line of the master on the spectral line of the stpectrum
    for (j=0;j<n3;j++)
    {
        wavelength_master_shift[j] = wavelength_master[j] + Delta_lambda(vrad_ccf,wavelength_master[j]); // calculate the position of the master spectrum line so its matches the line in the spectrum (shifted by ccf_vrad due to the BERV and a potential long-term binary)
    }

    //calculate_derivatives
    for (j=0;j<n3-1;j++)//n3-1 because the derivative can be calculated only for one point less
    {
        derivative_spectrum_master[j] = (spectrum_master[j+1] - spectrum_master[j]) / (wavelength_master_shift[j+1] - wavelength_master_shift[j]);
    }
    for (j=0;j<n3-2;j++)
    {
        derivative2_spectrum_master[j] = (derivative_spectrum_master[j+1] - derivative_spectrum_master[j]) / (wavelength_master_shift[j+1] - wavelength_master_shift[j]);
    }

    //obtain_same_sampling and difference between points in the high resolution master and the points in the spectrum
    obtain_same_sampling(wavelength,n2,wavelength_master_shift,n3,spectrum_master,wavelength_master_same,spectrum_master_same,index_same,diff); //wavelength_master_same,spectrum_master_same have the same sampling as wavelength,spectrum, to be able to calculate the Bouchy drift after.



    //linear_interpolation
    for (j=0;j<n2;j++)
    {
        if (diff[j] < 0)  //diff[i] = freq[i]-freq_oversampled[j] as defined in the function "obtain_same_sampling". diff < 0 => wavelength < wavelength_master_shift
        {
            x0 = wavelength_master_shift[index_same[j]-1];
            x1 = wavelength_master_shift[index_same[j]];
            y0 = spectrum_master[index_same[j]-1];
            y1 = spectrum_master[index_same[j]];
            y0_deriv = derivative_spectrum_master[index_same[j]-1];
            y1_deriv = derivative_spectrum_master[index_same[j]];
            y0_deriv2 = derivative2_spectrum_master[index_same[j]-1];
            y1_deriv2 = derivative2_spectrum_master[index_same[j]];
        }
        else
        {
            x0 = wavelength_master_shift[index_same[j]];
            x1 = wavelength_master_shift[index_same[j]+1];
            y0 = spectrum_master[index_same[j]];
            y1 = spectrum_master[index_same[j]+1];
            y0_deriv = derivative_spectrum_master[index_same[j]];
            y1_deriv = derivative_spectrum_master[index_same[j]+1];
            y0_deriv2 = derivative2_spectrum_master[index_same[j]];
            y1_deriv2 = derivative2_spectrum_master[index_same[j]+1];
        }
        slope = (y1-y0)/(x1-x0);
        ordinate_cross = y1 - slope*x1;
        slope_deriv = (y1_deriv-y0_deriv)/(x1-x0);
        ordinate_cross_deriv = y1_deriv - slope_deriv*x1;
        slope_deriv2 = (y1_deriv2-y0_deriv2)/(x1-x0);
        ordinate_cross_deriv2 = y1_deriv2 - slope_deriv2*x1;

        spectrum_master_same[j]             = slope*wavelength[j] + ordinate_cross;
        derivative_spectrum_master_same[j]  = slope_deriv*wavelength[j] + ordinate_cross_deriv;
        derivative2_spectrum_master_same[j] = slope_deriv2*wavelength[j] + ordinate_cross_deriv2;
        //printf("diff = %.4f\tw=%.4f\tw_master0=%.4f\tw_master1=%.4f\ty0=%.4f\ty1=%.4f\ty_interpol=%.4f\n",diff[j],wavelength[j],x0,x1,y0,y1,spectrum_master_same[j]);
    }

    //BOUCHY method using a linear least square.
    for (j=0;j<n2;j++)
    {
        sig_spectrum = pow(spectrum_raw[j] + pow(detector_noise,2),1/2.); //We assume that the noise on the master is negligible, therefore no noise on the derivative

        // Initializing the variables for the GSL linear lsq
        gsl_matrix_set (X, j, 0, spectrum_master_same[j]);
        gsl_matrix_set (X, j, 1, derivative_spectrum_master_same[j]);
        gsl_matrix_set (X, j, 2, derivative2_spectrum_master_same[j]);
        //gsl_matrix_set (X, j, 3, 1.0);
        gsl_vector_set (y, j, spectrum[j]);
        gsl_vector_set (w, j, 1.0/(sig_spectrum*sig_spectrum));
    }

    // Fitting linlsq with GSL, see http://www.gnu.org/software/gsl/manual/gsl-ref_37.html for more info on the GSL linlsq fit
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n2, nb_para);
    gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
    gsl_multifit_linear_free (work);


    //gsl_vector_get(c,(0)) = c
    //gsl_vector_get(c,(1)) = c . Dlambda
    //Dlambda = (c . Dlambda) / c
    //sig_Dlambda = sqrt((1/c)^2 * sig_cDlambda^2 + (-cDlambda/c^2)^2 * sig_c^2)
    //RV = -1. * Dlambda/lambda*c_lum + vrad_ccf #The -1 comes from testing. vrad_ccf because with shifted the entire spectrum by this value, so we have to add it
    //sig_RV = sig_Dlambda/lambda*c_lum
    // BE CAREFULL gsl_matrix_get(cov,1,1) = sigma_(c . Dlambda)**2 and gsl_matrix_get(cov,0,0) = sigma_c**2. Already squared, so BE CAREFULL WHEN CALCULATING THE ERROR

    *para0 = gsl_vector_get(c,(0));
    *para1 = gsl_vector_get(c,(1));
    *para2 = gsl_vector_get(c,(2));
    delta_lambda = gsl_vector_get(c,(1)) / gsl_vector_get(c,(0));
    //gsl_matrix_get(cov,1,1) equal to sigma**2, so no need to put the power 2
    sig_delta_lambda = pow(pow(1./gsl_vector_get(c,(0)),2)*gsl_matrix_get(cov,1,1) + pow(-gsl_vector_get(c,(1))/pow(gsl_vector_get(c,(0)),2),2)*gsl_matrix_get(cov,0,0),1/2.);
    printf("%.2f\n",-1. * delta_lambda / freq_line * CLUM);
    *RV = -1. * delta_lambda / freq_line * CLUM + vrad_ccf;
    *sig_RV = sig_delta_lambda / freq_line * CLUM;
    //*constant = gsl_vector_get(c,(3));
    *chi2 = chisq / (n2 - nb_para - 1);


    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);

//    free(wavelength_master_same);
//    free(derivative_spectrum_master);
//    free(derivative2_spectrum_master);
//    free(spectrum_master_same);
//    free(derivative_spectrum_master_same);
//    free(derivative2_spectrum_master_same);
    free(index_same);
    free(diff);
//    free(wavelength_master_shift);
}





static PyObject *get_RV_line_by_line(PyObject *self, PyObject *args)
{
	PyArrayObject *a, *b, *c, *d, *e, *f;
	PyArrayObject *aa, *bb, *cc, *dd;

	int i, j, n1, n2, n3, nb_fit_para=2;//, dim[1];
    double detector_noise;
    double *vrad_ccf,*wavelength,*spectrum_raw,*spectrum,*wavelength_master,*spectrum_master,*RV_vect,*sig_RV_vect,*chisq_vect;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|d", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &PyArray_Type, &e, &PyArray_Type, &f, &detector_noise))
    return NULL;

	vrad_ccf   = (double *) (a->data + 0*a->strides[0]);
	wavelength = (double *) (b->data + 0*b->strides[0]);
    spectrum_raw   = (double *) (c->data + 0*c->strides[0]);
    spectrum   = (double *) (d->data + 0*d->strides[0]);
    wavelength_master = (double *) (e->data + 0*e->strides[0]);
    spectrum_master   = (double *) (f->data + 0*f->strides[0]);
	n1 = (int)a->dimensions[0];
    n2 = (int)b->dimensions[0];
    n3 = (int)e->dimensions[0];

    RV_vect       = (double *) malloc(n1*sizeof(double));
    sig_RV_vect   = (double *) malloc(n1*sizeof(double));
    chisq_vect    = (double *) malloc(n1*sizeof(double));

    double **fit_para = (double **)malloc(sizeof(double *)*n1);
    for (i=0; i<n1; i++) fit_para[i] = (double *)malloc(sizeof(double)*nb_fit_para);

    get_RV_line_by_line_c(vrad_ccf, RV_vect, sig_RV_vect, chisq_vect, n1, wavelength, spectrum_raw, spectrum, n2, wavelength_master, spectrum_master, n3, detector_noise, fit_para);

	// dim[0] = n1;
	// aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // bb = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  // cc = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  npy_intp dim[1] = {n1};
	aa = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
	bb = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  cc = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);

	for (i = 0; i < n1; i++) {
        *(double *) (aa->data + i*aa->strides[0]) = RV_vect[i];
        *(double *) (bb->data + i*bb->strides[0]) = sig_RV_vect[i];
        *(double *) (cc->data + i*cc->strides[0]) = chisq_vect[i];
    }
    // int dimensions[2]={n1,nb_fit_para};
    // dd = (PyArrayObject *) PyArray_FromDims(2, dimensions, PyArray_DOUBLE);
    npy_intp dim2[2] = {n1,nb_fit_para};
    dd = (PyArrayObject *) PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
    for (i=0; i<n1; i++) for (j=0; j<nb_fit_para; j++)
        *(double *)(dd->data + i*dd->strides[0] + j*dd->strides[1]) = fit_para[i][j];

    for (j=0; j<n1; j++) free(fit_para[j]);
    free(fit_para);
    free(RV_vect);
    free(sig_RV_vect);
    free(chisq_vect);

    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
	return Py_BuildValue("NNNN",aa,bb,cc,dd);
}

static PyObject *get_RV_line_by_line2(PyObject *self, PyObject *args)
{
    PyArrayObject *a, *b, *c, *d, *e;

    int n1, n2, n3;
    double detector_noise,vrad_ccf,normalization=0,RV=100000,sig_RV=100000,constant=0,chi2=100000,freq_line=100000;
    double *wavelength,*spectrum_raw,*spectrum,*wavelength_master,*spectrum_master;


    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|ddd", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &PyArray_Type, &e, &detector_noise, &vrad_ccf, &freq_line))
    return NULL;

    wavelength = (double *) (a->data + 0*a->strides[0]);
    spectrum_raw   = (double *) (b->data + 0*b->strides[0]);
    spectrum   = (double *) (c->data + 0*c->strides[0]);
    wavelength_master = (double *) (d->data + 0*d->strides[0]);
    spectrum_master   = (double *) (e->data + 0*e->strides[0]);
    n1 = (int)a->dimensions[0];
    n2 = (int)b->dimensions[0];
    n3 = (int)e->dimensions[0];

    get_RV_line_by_line_Bouchy_without_offset(&normalization, &RV, &sig_RV, &constant, &chi2, freq_line, wavelength, spectrum_raw, spectrum, n2, wavelength_master, spectrum_master, n3, detector_noise,vrad_ccf);

    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
    return Py_BuildValue("ddddd",normalization, RV, sig_RV, constant, chi2);
}

static PyObject *get_RV_line_by_line2_with_offset(PyObject *self, PyObject *args)
{
    PyArrayObject *a, *b, *c, *d, *e;

    int n1, n2, n3;
    double detector_noise,vrad_ccf,normalization=0,RV=100000,sig_RV=100000,constant=0,chi2=100000,freq_line=100000;
    double *wavelength,*spectrum_raw,*spectrum,*wavelength_master,*spectrum_master;


    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|ddd", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &PyArray_Type, &e, &detector_noise, &vrad_ccf, &freq_line))
    return NULL;

    wavelength = (double *) (a->data + 0*a->strides[0]);
    spectrum_raw   = (double *) (b->data + 0*b->strides[0]);
    spectrum   = (double *) (c->data + 0*c->strides[0]);
    wavelength_master = (double *) (d->data + 0*d->strides[0]);
    spectrum_master   = (double *) (e->data + 0*e->strides[0]);
    n1 = (int)a->dimensions[0];
    n2 = (int)b->dimensions[0];
    n3 = (int)e->dimensions[0];

    get_RV_line_by_line_Bouchy_with_offset(&normalization, &RV, &sig_RV, &constant, &chi2, freq_line, wavelength, spectrum_raw, spectrum, n2, wavelength_master, spectrum_master, n3, detector_noise,vrad_ccf);

    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
    return Py_BuildValue("ddddd",normalization, RV, sig_RV, constant, chi2);
}


static PyObject *get_RV_line_by_line3(PyObject *self, PyObject *args)
{
    PyArrayObject *a, *b, *c, *d, *e, *aa, *bb, *cc, *dd, *ee, *ff;

    int i, n1, n2, n3;//, dim1[1], dim[1];
    double detector_noise,vrad_ccf,para0=0,para1=0,para2=0,RV=100000,sig_RV=100000,constant=0,chi2=100000,freq_line=100000;
    double *wavelength,*spectrum_raw,*spectrum,*wavelength_master,*spectrum_master;
    double *spectrum_master_same,*derivative_spectrum_master_same,*derivative2_spectrum_master_same;
    double *wavelength_master_shift, *derivative_spectrum_master, *derivative2_spectrum_master;


    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|ddd", &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c, &PyArray_Type, &d, &PyArray_Type, &e, &detector_noise, &vrad_ccf, &freq_line))
    return NULL;

    wavelength = (double *) (a->data + 0*a->strides[0]);
    spectrum_raw   = (double *) (b->data + 0*b->strides[0]);
    spectrum   = (double *) (c->data + 0*c->strides[0]);
    wavelength_master = (double *) (d->data + 0*d->strides[0]);
    spectrum_master   = (double *) (e->data + 0*e->strides[0]);
    n1 = (int)a->dimensions[0];
    n2 = (int)b->dimensions[0];
    n3 = (int)e->dimensions[0];

    wavelength_master_shift     = (double *) malloc(n3*sizeof(double));
    derivative_spectrum_master  = (double *) malloc(n3*sizeof(double));
    derivative2_spectrum_master = (double *) malloc(n3*sizeof(double));

    spectrum_master_same             = (double *) malloc(n2*sizeof(double));
    derivative_spectrum_master_same  = (double *) malloc(n2*sizeof(double));
    derivative2_spectrum_master_same = (double *) malloc(n2*sizeof(double));

    //init
    for (i=0;i<n3;i++)
    {
        wavelength_master_shift[i] = 0.0;
        derivative_spectrum_master[i] = 0.0;
        derivative2_spectrum_master[i] = 0.0;
    }
    for (i=0;i<n2;i++)
    {
        spectrum_master_same[i] = 0.0;
        derivative_spectrum_master_same[i] = 0.0;
        derivative2_spectrum_master_same[i] = 0.0;
    }

    get_RV_line_by_line_Bouchy_with_offset_second_order_deriv(&para0, &para1, &para2, &RV, &sig_RV, &constant, &chi2, freq_line, wavelength, spectrum_raw, spectrum, n2, wavelength_master, spectrum_master, n3, detector_noise,vrad_ccf, wavelength_master_shift, derivative_spectrum_master, derivative2_spectrum_master,spectrum_master_same,derivative_spectrum_master_same,derivative2_spectrum_master_same);

    // dim1[0] = n3;
    // dim[0] = n2;
    // dd = (PyArrayObject *) PyArray_FromDims(1, dim1, PyArray_DOUBLE);
    // ee = (PyArrayObject *) PyArray_FromDims(1, dim1, PyArray_DOUBLE);
    // ff = (PyArrayObject *) PyArray_FromDims(1, dim1, PyArray_DOUBLE);
    // aa = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    // bb = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    // cc = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
    npy_intp dim[1] = {n2};
    npy_intp dim1[1] = {n3};
  	aa = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  	bb = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    cc = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_DOUBLE);
    dd = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    ee = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    ff = (PyArrayObject *) PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    for (i = 0; i < n3; i++)
    {
        *(double *) (dd->data + i*dd->strides[0]) = wavelength_master_shift[i];
        *(double *) (ee->data + i*ee->strides[0]) = derivative_spectrum_master[i];
        *(double *) (ff->data + i*ff->strides[0]) = derivative2_spectrum_master[i];
    }
    for (i = 0; i < n2; i++)
    {
        *(double *) (aa->data + i*aa->strides[0]) = spectrum_master_same[i];
        *(double *) (bb->data + i*bb->strides[0]) = derivative_spectrum_master_same[i];
        *(double *) (cc->data + i*cc->strides[0]) = derivative2_spectrum_master_same[i];
    }

//    free(wavelength_master_shift);
//    free(derivative_spectrum_master);
//    free(derivative2_spectrum_master);
//    free(spectrum_master_same);
//    free(derivative_spectrum_master_same);
//    free(derivative2_spectrum_master_same);

    /*ATTENTION mettre N et non pas O car O incremente de 1 a chaque fois ce qui ne kill pas la memoire*/
    return Py_BuildValue("dddddddNNNNNN",para0, para1, para2, RV, sig_RV, constant, chi2, dd, ee, ff, aa, bb, cc);

}


static PyMethodDef calculate_RV_line_by_line_methods[] = {

	{"get_RV_line_by_line", get_RV_line_by_line, METH_VARARGS,
 	"calculate spectrum"},
    {"get_RV_line_by_line2_with_offset", get_RV_line_by_line2_with_offset, METH_VARARGS,
        "calculate spectrum"},
    {"get_RV_line_by_line2", get_RV_line_by_line2, METH_VARARGS,
        "calculate spectrum"},
    {"get_RV_line_by_line3", get_RV_line_by_line3, METH_VARARGS,
        "calculate spectrum"},
    {NULL, NULL, 0, NULL}

};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "calculate_RV_line_by_line3",     /* m_name */
    " ",                   /* m_doc */
    -1,                    /* m_size */
    calculate_RV_line_by_line_methods,   /* m_methods */
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
    m = Py_InitModule3("calculate_RV_line_by_line", calculate_RV_line_by_line_methods," ");
    #endif

    if (m == NULL) return NULL;

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_calculate_RV_line_by_line3(void)
{
    import_array();
    return moduleinit();
}
#else
PyMODINIT_FUNC initcalculate_RV_line_by_line(void)
{
    (void) moduleinit();

    import_array();
}
#endif


#undef CLUM
