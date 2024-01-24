/*
 MD.c - a simple molecular dynamics program for simulating real gas properties of Lennard-Jones particles.
 
 Copyright (C) 2016  Jonathan J. Foley IV, Chelsea Sweet, Oyewumi Akinfenwa
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 Electronic Contact:  foleyj10@wpunj.edu
 Mail Contact:   Prof. Jonathan Foley
 Department of Chemistry, William Paterson University
 300 Pompton Road
 Wayne NJ 07470
 
 */
#include "MDCuda.h"

// --------------------------CUDA --------------------------

#define NUM_THREADS_PER_BLOCK 512

// --------------------------CUDA --------------------------

// Number of particles
int N = 5000;

// --------------------------CUDA --------------------------

__device__ int N_Cuda;
__device__ double PE_Cuda;

// --------------------------CUDA --------------------------

double PEG;
double NA = 6.022140857e23;
double kBSI = 1.38064852e-23;  // m^2*kg/(s^2*K)

//  Size of box, which will be specified in natural units
double L;

//  Initial Temperature in Natural Units
double Tinit;  //2;

//  Vectors!
const int MAXPART=5001;
//  Position
double* r = (double *) malloc(MAXPART*3*sizeof(double));
//  Velocity
double* v= (double *) malloc(MAXPART*3*sizeof(double));
//  Acceleration
double* a= (double *) malloc(MAXPART*3*sizeof(double));

// --------------------------CUDA --------------------------

double *r_Cuda, *a_Cuda; 
double aux = N * 3 * sizeof(double);

// --------------------------CUDA --------------------------

char *atype = (char *)malloc(3 * sizeof(char));

//  Function prototypes

//  initialize positions on simple cubic lattice, also calls function to initialize velocities
void initialize();  

//  update positions and velocities using Velocity Verlet algorithm 
//  print particle coordinates to file for rendering via VMD or other animation software
//  return 'instantaneous pressure'
double VelocityVerlet(double dt, FILE *fp);  

//  Compute Force using F = -dV/dr
//  solve F = ma for use in Velocity Verlet
//  Compute total potential energy from particle coordinates
__global__ void computeAccelerationsGPU(double *a_Cuda, double *r_Cuda, double *Pot_Cuda);

//  Numerical Recipes function for generation gaussian distribution
double gaussdist();

//  Initialize velocities according to user-supplied initial Temperature (Tinit)
void initializeVelocities();

//  Compute mean squared velocity from particle velocities and total kinetic energy from particle mass and velocities
double MeanSquaredVelocityKin();


void computeAccelerations();

int main(){

    int i, NumTime;
    double dt, Vol, Temp, Press, Pavg = 0, Tavg = 0, rho, VolFac, TempFac, PressFac, timefac, KE, mvs, gc, Z;
    char prefix[1000], tfn[1000], ofn[1000], afn[1000];
    FILE *tfp, *ofp, *afp;
    
    scanf("%s",prefix);
    strcpy(tfn,prefix);
    strcat(tfn,"_traj.xyz");
    strcpy(ofn,prefix);
    strcat(ofn,"_output.txt");
    strcpy(afn,prefix);
    strcat(afn,"_average.txt");
    
    scanf("%s",atype);
    
    if (strcmp(atype,"He")==0) {
        
        VolFac = 1.8399744000000005e-29;
        PressFac = 8152287.336171632;
        TempFac = 10.864459551225972;
        timefac = 1.7572698825166272e-12;
        
    }
    else if (strcmp(atype,"Ne")==0) {
        
        VolFac = 2.0570823999999997e-29;
        PressFac = 27223022.27659913;
        TempFac = 40.560648991243625;
        timefac = 2.1192341945685407e-12;
        
    }
    else if (strcmp(atype,"Ar")==0) {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        
    }
    else if (strcmp(atype,"Kr")==0) {
        
        VolFac = 4.5882712000000004e-29;
        PressFac = 59935428.40275003;
        TempFac = 199.1817584391428;
        timefac = 8.051563913585078e-13;
        
    }
    else if (strcmp(atype,"Xe")==0) {
        
        VolFac = 5.4872e-29;
        PressFac = 70527773.72794868;
        TempFac = 280.30305642163006;
        timefac = 9.018957925790732e-13;
        
    }
    else {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        strcpy(atype,"Ar");
        
    }

    scanf("%lf",&Tinit);

    if (Tinit<0.) {
        printf("\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!\n");
        exit(0);
    }

    if (N>=MAXPART) {
        printf("\n\n\n  MAXIMUM NUMBER OF PARTICLES IS %i\n\n  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY \n\n", MAXPART);
        exit(0);
    }

    // Convert initial temperature from kelvin to natural units
    Tinit /= TempFac;
    
    scanf("%lf",&rho);

    // Copy N to the device variable N_Cuda
    cudaMemcpyToSymbol(N_Cuda, &N, sizeof(int));

    Vol = N/(rho*NA);
    
    Vol /= VolFac;

    if (Vol<N) {
        printf("\n\n\n  YOUR DENSITY IS VERY HIGH!\n\n");
        printf("  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n",N,Vol);
        printf("  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE\n");
        printf("  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n\n");
        exit(0);
    }

    // Length of the box in natural units:
    L = cbrt(Vol);
    
    //  Files that we can write different quantities to
    tfp = fopen(tfn,"w");    //  The MD trajectory, coordinates of every particle at each timestep
    ofp = fopen(ofn,"w");    //  Output of other quantities (T, P, gc, etc) at every timestep
    afp = fopen(afn,"w");    //  Average T, P, gc, etc from the simulation
    
    NumTime = 200;
    dt = 0.5e-14/timefac;

    if (strcmp(atype,"He")==0) {
        dt = 0.2e-14/timefac;
        NumTime=50000;
    }

    initialize();
    
    computeAccelerations();


    fprintf(tfp,"%i\n",N);
    fprintf(ofp,"  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n");

    for (i=0; i<NumTime+1; i++) {
        
        Press = VelocityVerlet(dt, tfp);
        Press *= PressFac;

        mvs = MeanSquaredVelocityKin()/N;
        KE = MeanSquaredVelocityKin()*0.5;

        Temp = mvs/3 * TempFac;

        gc = NA*Press*(Vol*VolFac)/(N*Temp);
        Z  = Press*(Vol*VolFac)/(N*kBSI*Temp);
        
        Tavg += Temp;
        Pavg += Press;
         fprintf(ofp,"  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n",i*dt*timefac,Temp,Press,KE, PEG, KE+PEG);
        //fprintf(ofp,"  %12.8e  %12.8f  %12.8f %12.8f  %12.8f  %12.8f \n",i*dt*timefac,Temp,Press,KE, PEG, KE+PEG);
    }
    
    // Because we have calculated the instantaneous temperature and pressure,
    // we can take the average over the whole simulation here
    Pavg /= NumTime;
    Tavg /= NumTime;
    Z = Pavg*(Vol*VolFac)/(N*kBSI*Tavg);
    gc = NA*Pavg*(Vol*VolFac)/(N*Tavg);

/*
    fprintf(afp,"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp," --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp,"  %12.12e  %12.12f       %12.12f     %12.12f       %12.12f        %12.12e         %i\n",i*dt*timefac,Tavg,Pavg,gc,Z,Vol*VolFac,N);
  */  

    fprintf(afp,"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp," --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp,"  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n",i*dt*timefac,Tavg,Pavg,gc,Z,Vol*VolFac,N);
    
    printf("\n  AVERAGE TEMPERATURE (K):                 %15.5f\n",Tavg);
    printf("\n  AVERAGE PRESSURE  (Pa):                  %15.5f\n",Pavg);
    printf("\n  PV/nT (J * mol^-1 K^-1):                 %15.5f\n",gc);
    printf("\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.5f\n",100*fabs(gc-8.3144598)/8.3144598);
    printf("\n  THE COMPRESSIBILITY (unitless):          %15.5f \n",Z);
    printf("\n  TOTAL VOLUME (m^3):                      %10.5e \n",Vol*VolFac);
    printf("\n  NUMBER OF PARTICLES (unitless):          %i \n", N);
    
    free(r);
    free(v);
    free(a);
    free(atype);
    fclose(tfp);
    fclose(ofp);
    fclose(afp);
    
    return 0;
}

void initialize() {
    int n, p=0 , i, j, k;
    double pos,pos1, posm, pos2;
    
    // Number of atoms in each direction
    n = int(ceil(cbrt(N)));
    
    //  spacing between atoms along a given direction
    pos = L / n;
    posm = pos*0.5;
    
    //  index for number of particles assigned positions

    //  initialize positions
    for (i=0; i<n; i++) {
        pos1 = posm+i*pos ;
        for (j=0; j<n; j++) {
            pos2 = j*pos + posm;
            for (k=0; k<n; k++) {
                if (p<N*3) {
                    r[p++] = pos1;
                    r[p++] = pos2;
                    r[p++] = k*pos + posm;
                }
            }
        }
    }
    
    // Call function to initialize velocities
    initializeVelocities();
}

//  Function to calculate the averaged velocity squared
double MeanSquaredVelocityKin() { 
    
    double vaux = 0;
    
    for (int i=0; i<N*3; i++) {
        vaux += v[i]*v[i];
    }
    
    return vaux;
}

// --------------------------CUDA --------------------------



//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
__global__ void computeAccelerationsGPU(double *a_Cuda, double *r_Cuda, double *Pot_Cuda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double sharedRk[NUM_THREADS_PER_BLOCK * 3];

    // Each thread loads the values of rk into shared memory
    for (int k = 0; k < 3; ++k) {
        sharedRk[threadIdx.x * 3 + k] = r_Cuda[i * 3 + k];
    }

    if (i < N_Cuda) {
        double local_VPot = 0.0;
        double v_CudaAux[3] = {0.0, 0.0, 0.0};

        for (int j = 0; j < N_Cuda; j++) {
            if (i != j) {

                double vals[3];
                double rij[3];
                double rSqd = 0;

                rij[0] = sharedRk[threadIdx.x * 3] - r_Cuda[j * 3];
                rij[1] = sharedRk[threadIdx.x * 3 + 1] - r_Cuda[j * 3 + 1];
                rij[2] = sharedRk[threadIdx.x * 3 + 2] - r_Cuda[j * 3 + 2];

                rSqd = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

                double rSqd3 = rSqd*rSqd*rSqd;
                double rSqd6 = rSqd3*rSqd3;
                local_VPot+=((1-rSqd3)/(rSqd6));
            
                double f = ((48 - 24*rSqd3)/(rSqd6*rSqd));

                vals[0] = rij[0] * f; 
                vals[1] = rij[1] * f; 
                vals[2] = rij[2] * f;

                v_CudaAux[0] += vals[0];
                v_CudaAux[1] += vals[1];
                v_CudaAux[2] += vals[2];
            }
        }
        Pot_Cuda[i] = local_VPot;

        a_Cuda[i * 3] = v_CudaAux[0];
        a_Cuda[i * 3 + 1] = v_CudaAux[1];
        a_Cuda[i * 3 + 2] = v_CudaAux[2];
    
    }
}

void computeAccelerations() {

    double Pot = 0.;
    double v_Pot[N];
    double* Pot_Cuda;

    int siz = N*3; 

    for (int i = 0; i < siz; i++) { 
        a[i] = 0;
    }

    for (int i = 0; i < N; i++)
        v_Pot[i] = 0;

    cudaMalloc((void**)&r_Cuda, aux);
    cudaMalloc((void**)&a_Cuda, aux);
    cudaMalloc((void**)&Pot_Cuda, N * sizeof(double));
    checkCUDAError("Mem Allocation");

    cudaMemcpy(a_Cuda, a, aux, cudaMemcpyHostToDevice);
    cudaMemcpy(r_Cuda, r, aux, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy Host -> Device");

    int bpg = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;  // Arredondamento para cima

    computeAccelerationsGPU<<<bpg, NUM_THREADS_PER_BLOCK>>>(a_Cuda, r_Cuda, Pot_Cuda);
    cudaDeviceSynchronize();
    checkCUDAError("Error in computeAccelerationsGPU");

    cudaMemcpy(a, a_Cuda, aux, cudaMemcpyDeviceToHost);


    cudaMemcpy(v_Pot, Pot_Cuda, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkCUDAError("Memcpy Device -> Host");

    for (int i = 0; i < N; i++) {
        Pot += v_Pot[i];
    }

    cudaFree(r_Cuda);
    cudaFree(a_Cuda);
    cudaFree(Pot_Cuda);
    checkCUDAError("Free Mem");

    PEG= Pot*4;
}

// --------------------------CUDA --------------------------

double VelocityVerlet(double dt, FILE *fp) {
    
    int i;
    double psum = 0., temp1, temp2, dt1 = 0.5 * dt;

    for (i=0; i<N*3; i += 2) {
        temp1 = a[i] * dt1;
        r[i] += (v[i] + temp1) * dt;
        v[i] += temp1;

        temp2 = a[i+1] * dt1;
        r[i+1] += (v[i+1] + temp2) * dt;
        v[i+1] += temp2;
    }

    computeAccelerations();

    for (i=0; i<N*3; i += 2) {
        v[i] += a[i] * dt1;
        v[i+1] += a[i+1] * dt1;
    }
    
    // Elastic walls
    for (i=0; i<N*3; i += 2) {
        if (r[i]<0. || r[i]>=L) {
            v[i] *=-1.;
            psum += fabs(v[i]);
        }

        if (r[i+1]<0. || r[i+1]>=L) {
            v[i+1] *=-1.;
            psum += fabs(v[i+1]);
        }
    }
    
    return psum/(3*L*L*dt);
}

void initializeVelocities() {
    
    int i,j;
    for ( i=0; i < N*3; i += 2) {
        v[i] = gaussdist();
        v[i+1] = gaussdist();
    }
    
    double vCM[3] = {0, 0, 0};

    for ( i=0; i<N; i++) {
        for ( j=0; j<3; j++) {
            vCM[j] += v[i*3+j];
        }
    }

    for (i=0; i<3; i++) vCM[i] /= N;
    
     for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            v[i*3+j] -= vCM[j];
        }
    }

    double vSqdSum, lambda;
    vSqdSum=0.;
    for (int i = 0; i < N * 3; i += 5) {
        vSqdSum += v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2] + v[i + 3] * v[i + 3] + v[i + 4] * v[i + 4];
    }
    
    lambda = sqrt( 3*(N-1)*Tinit/vSqdSum);
    
    for (int i=0; i<N*3; i +=2) {
        v[i] *= lambda;
        v[i+1] *= lambda;
    }
}

//  Numerical recipes Gaussian distribution number generator
double gaussdist() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;
        
        return v2*fac;
    } else {
        
        available = false;
        return gset;
        
    }
}