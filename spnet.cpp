// SPNET: Spiking neural network with axonal conduction delays and STDP
// Created by Eugene M. Izhikevich, May 17, 2004, San Diego, CA
// Saves spiking data each second in file spikes.dat
// To plot spikes, use MATLAB code: load spikes.dat;plot(spikes(:,1),spikes(:,2),'.');
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tclap/CmdLine.h>
#include <queue>

#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1
using namespace std;
using namespace TCLAP;

class SpikingNetwork {

private:
  //int Ne;		// excitatory neurons			
  //int Ni;		// inhibitory neurons				 
  int N;		// total number of neurons	
  int M;		// the number of synapses per neuron 
  int D;		// maximal axonal conduction delay
  int numClasses;       // number of dynamical classes
  //  float sm;		                // maximal synaptic strength		
  int *post;	// post[N][M];				// indeces of postsynaptic neurons
  float *s, *sd; //s[N][M], sd[N][M];		        // matrix of synaptic weights and their derivatives
  short *delays_length; //[N][D];	                // distribution of delays
  short *delays; //[N][D][M];
  //^^[index of presynaptic neuron][delay to efferent][column of post[N][] containing the index of efferent for neuron N]
  int *N_pre, *I_pre, *D_pre; //N_pre[N], I_pre[N][3*M], D_pre[N][3*M];	// presynaptic information
  float **s_pre, **sd_pre; // [N][3*M];	// presynaptic weights
  float *LTP, *LTD; // LTP[N][frequency+1+D], LTD[N];	                // STDP functions 
  int *count; //class instance counts [numClasses]
  int *unitClass; // [N] classIDs for each neuron
  float  *C, *kdyn, *vr, *vt, *peak, *a, *b, *bhyp, *c, *d, *umax, *caInact;	//[numClasses] (neuronal dynamics parameters)
  bool *plastic, *record;
  float *A_plus, *A_minus, *tau_plus, *tau_minus, *max_weight; //[numClasses] class-level STDP params
  float *Cinv, *vrPlusVt, *kVrVt, *ab, *abVr, *LTPdecay, *LTDdecay;   // [numClasses] calculated coefficients
  float *v, *u; //[N]				// activity variables
  //	double	C_max=10;		
  //	static const	int	W=3;	                        // initial width of polychronous groups
  //	int     min_group_path = 7;		        // minimal length of a group
  //	int	min_group_time = 40;	                // minimal duration of a group (ms)

  //	static const	int	latency = D; // maximum latency 
  //--------------------------------------------------------------
  //	static const int	polylenmax = N;

public:
  //SpikingNetwork(int Ne, int Ni, int M, int D);
  SpikingNetwork(string filename);
  SpikingNetwork();
  void simulate(int maxSecs, int trainSecs, int testSecs, string inFileName, float scale,
		bool kaldiMode, int numFeats, int stepSize, string outFileName, int frequency);
  void saveTo(string filename);
  //	void polychronous(int nnum);
  //	void all_polychronous();
};

//default network initialization
SpikingNetwork::SpikingNetwork() {
  int i, j, k, jj, dd, exists, r;
  N = 1000;
  M = 100;
  numClasses = 2;
  D = 20;
  post = new int[N * M];
  s = new float[N * M];
  sd = new float[N * M];
  delays_length = new short[N * D];
  delays = new short[N * D * M];
  N_pre = new int[N];
  I_pre = new int[N * 3 * M];
  D_pre = new int[N * 3 * M];
  s_pre = new float*[N * 3 * M];
  sd_pre = new float*[N * 3 * M];
  unitClass = new int[N];

  count = new int[numClasses];
  C = new float[numClasses];
  kdyn = new float[numClasses];
  vr = new float[numClasses];
  vt = new float[numClasses];
  peak = new float[numClasses];
  a = new float[numClasses];
  b = new float[numClasses];
  bhyp = new float[numClasses];
  c = new float[numClasses];
  d = new float[numClasses];
  umax = new float[numClasses];
  caInact = new float[numClasses];
  A_plus = new float[numClasses];
  A_minus = new float[numClasses];
  tau_plus = new float[numClasses];
  tau_minus = new float[numClasses];
  plastic = new bool[numClasses];  
  max_weight = new float[numClasses];
  record = new bool[numClasses];
  
  Cinv = new float[numClasses];
  vrPlusVt = new float[numClasses];
  kVrVt = new float[numClasses];
  ab = new float[numClasses];
  abVr = new float[numClasses];
  LTPdecay = new float[numClasses];
  LTDdecay = new float[numClasses];
    
  //RS neurons from Izhikevich 2007, "Dynamical Systems in Neuroscience", Chapter 8
  count[0] = 800;
  C[0] = 100.0;
  kdyn[0] = 0.7;
  vr[0] = -60.0;
  vt[0] = -40.0;
  peak[0] = 35.0;
  a[0] = 0.03; 
  b[0] = -2.0;
  bhyp[0] = -2.0;
  c[0] = -50.0;
  d[0] = 100.0;
  umax[0] = 10000.0;
  caInact[0] = -300.0;
  A_plus[0] = 0.10;
  A_minus[0] = 0.12;
  tau_plus[0] = 20.0;
  tau_minus[0] = 20.0;
  plastic[0] = true;
  max_weight[0] = 10.0;
  record[0] = true;
  
  //FS (basket) neurons from Izhikevich 2008
  count[1] = 200;
  C[1] = 20.0;
  kdyn[1] = 1.0;
  vr[1] = -55;
  vt[1] = -40;
  peak[1] = 25.0;
  a[1] = 0.15;
  b[1] = 8;
  bhyp[1] = 8;
  c[1] = -55.0;
  d[1] = 200.0;
  umax[1] = 10000.0;
  caInact[1] = -300.0;
  A_plus[1] = 0.0;
  A_minus[1] = 0.0;
  tau_plus[1] = 1.0;
  tau_minus[1] = 1.0;
  plastic[1] = false;
  max_weight[1] = -5.0;
  record[1] = true;
  
  for (i = 0; i < count[0]; i++) {
    unitClass[i] = 0;	// RS type 
  }
  for (i = count[0]; i < N; i++) {
    unitClass[i] = 1; // FS type
  }
  for (i = 0; i < numClasses; i++){
    Cinv[i] = 1/C[i];
    vrPlusVt[i] = vr[i] + vt[i];
    kVrVt[i] = kdyn[i] * vr[i] * vt[i];
    ab[i] = a[i] * b[i];
    abVr[i] = a[i] * b[i] * vr[i];
    LTPdecay[i] = 1.0 - (1.0/tau_plus[i]); 
    LTDdecay[i] = 1.0 - (1.0/tau_minus[i]); 
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      do {
	exists = 0;		// avoid multiple synapses
	if (i < count[0]) 
	  r = getrandom(N);
	else
	  r = getrandom(count[0]);		// inh -> exc only
	if (r == i)
	  exists = 1;				      	// no self-synapses 
	for (k = 0; k < j; k++)
	  if (post[i * M + k] == r)
	    exists = 1;	// synapse already exists  
      } while (exists == 1);
      post[i * M + j] = r;
    }
  }
  for (i = 0; i < count[0]; i++) { 
    for (j = 0; j < M; j++) {
      s[i * M + j] = 6.0; // initial exc. synaptic weights
    }
  }
  for (i = count[0]; i < N; i++) {
    for (j = 0; j < M; j++) {
      s[i * M + j] = -5.0; // inhibitory synaptic weights
    }
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      sd[i * M + j] = 0.0; // synaptic derivatives 
    }
  }
  for (i = 0; i < N; i++) {
    short ind = 0;
    if (i < count[0]) { 
      for (j = 0; j < D; j++) {
	delays_length[i * D + j] = M / D;// uniform distribution of exc. synaptic delays
	for (k = 0; k < delays_length[i * D + j]; k++) {
	  delays[i * D * M + j * M + k] = ind++;
	}
      }
    } else {
      for (j = 0; j < D; j++) {
	delays_length[i * D + j] = 0;
      }
      delays_length[i * D + 0] = M;	// all inhibitory delays are 1 ms
      for (k = 0; k < delays_length[i * D + 0]; k++) {
	delays[i * D * M + 0 * M + k] = ind++;
      }
    }
  }

  for (i = 0; i < N; i++) {
    N_pre[i] = 0;
    for (j = 0; j < count[0]; j++) {
      for (k = 0; k < M; k++) {
	if (post[j * M + k] == i) {		// find all presynaptic neurons 
	  I_pre[i * 3 * M + N_pre[i]] = j;// add this neuron to the list
	  for (dd = 0; dd < D; dd++)	// find the delay
	    for (jj = 0; jj < delays_length[j * D + dd]; jj++)
	      if (post[j * M + delays[j * D * M + dd * M + jj]]
		  == i)
		D_pre[i * 3 * M + N_pre[i]] = dd;
	  s_pre[i * 3 * M + N_pre[i]] = &s[j * M + k];// pointer to the synaptic weight	
	  sd_pre[i * 3 * M + N_pre[i]++] = &sd[j * M + k];// pointer to the derivative
	}
      }
    }
  }
}

void SpikingNetwork::saveTo(string filename){
  cout << "save to\n";
  ofstream saveFile;
  saveFile.open(filename);
  if(!saveFile.fail()){
    saveFile << to_string(numClasses) + "," + to_string(N) + "," + 
      to_string(M) + "," + to_string(D) + "\n";
    int i,j,k;
    //build temporary delay index
    short* temp_D = new short[N*M];
    for (i=0; i < N; i++){
      for (j=0; j < D; j++){
	for (k=0; k < delays_length[i * D + j]; k++){
	  temp_D[i * M + delays[i * D * M + j * M + k]] = j;
	}
      }
    }
    //write class-level params
    for (i = 0; i < numClasses; i++){
      saveFile << to_string(count[i]) + "," +
	to_string(C[i]) + "," +
	to_string(kdyn[i]) + "," +
	to_string(vr[i]) + "," +
	to_string(vt[i]) + "," +
	to_string(peak[i]) + "," +
	to_string(a[i]) + "," +
	to_string(b[i]) + "," +
	to_string(bhyp[i]) + "," +
	to_string(c[i]) + "," +
	to_string(d[i]) + "," +
	to_string(umax[i]) + "," +
	to_string(caInact[i]) + "," +
	to_string(A_plus[i]) + "," +
	to_string(A_minus[i]) + "," +
	to_string(tau_plus[i]) + "," +
	to_string(tau_minus[i]) + "," +
	to_string((uint)plastic[i]) + "," +
	to_string(max_weight[i]) + "," +
	to_string((uint)record[i]) +"\n";
    }
    //write neuron-level parameters
    for (i=0; i < N; i++){
      saveFile << to_string(unitClass[i]) + "\n";
    }
    //write synapse-level parameters
    for (i=0; i < N; i++){
      for (j=0; j < M; j++){
	saveFile << to_string(post[i*M+j]) + "," + to_string(s[i*M+j]) +
	  "," + to_string(sd[i*M+j]) + "," + to_string(temp_D[i*M+j]+1) + ";";
      }
      saveFile << "\n";
    }
    delete [] temp_D;
  }
  saveFile.close();
}

SpikingNetwork::SpikingNetwork(string filename){
  //sm = 10.0;
  cout << "file constructor\n";
  ifstream infile;
  infile.open(filename);
  if (!infile.fail()){
    cout << "file opened\n";
    string line;
    getline(infile, line);
    size_t next = 0;
    size_t last = 0;
    int i,j,k,jj,dd;
    int input;
    //get metaparams
    cout << "get metaparams\n";
    for (i = 0; i < 4; i++){
      next = line.find(",", last);
      input = stoi(line.substr(last, next - last));
      switch(i){
      case 0:
	numClasses = input;
	break;
      case 1:
	N = input;
	break;
      case 2:
	M = input;
	break;
      case 3:
	D = (short)input;
	break;
      }
      last = next + 1;
    }
    //get class-level params
    cout << "get class params\n";
    count = new int[numClasses];
    C = new float[numClasses];
    kdyn = new float[numClasses];
    vr = new float[numClasses];
    vt = new float[numClasses];
    peak = new float[numClasses];
    a = new float[numClasses];
    b = new float[numClasses];
    bhyp = new float[numClasses];
    c = new float[numClasses];
    d = new float[numClasses];
    umax = new float[numClasses];
    caInact = new float[numClasses];
    A_plus = new float[numClasses];
    A_minus = new float[numClasses];
    tau_plus = new float[numClasses];
    tau_minus = new float[numClasses];
    plastic = new bool[numClasses];  
    max_weight = new float[numClasses];
    record = new bool[numClasses];
    
    float flput;
    bool bput;
    for (i = 0; i < numClasses; i++){
      getline(infile, line);
      for (j = 0; j < 20; j++){
	next = line.find(",", last);
	if (j == 0) {
	  input = stoi(line.substr(last, next - last));
	} else if (j == 17 || j == 19){
	  bput = (bool)stoul(line.substr(last, next - last));
	} else {
	  flput = stof(line.substr(last, next - last));
	}  
	switch(j){
	case 0:
	  count[i] = input;
	  break;
	case 1:
	  C[i] = flput;
	  break;
	case 2:
	  kdyn[i] = flput;
	  break;
	case 3:
	  vr[i] = flput;
	  break;
	case 4:
	  vt[i] = flput;
	  break;
	case 5:
	  peak[i] = flput;
	  break;
	case 6:
	  a[i] = flput;
	  break;
	case 7:
	  b[i] = flput;
	  break;
	case 8:
	  bhyp[i] = flput;
	  break;
	case 9:
	  c[i] = flput;
	  break;
	case 10:
	  d[i] = flput;
	  break;
	case 11:
	  umax[i] = flput;
	  break;
	case 12:
	  caInact[i] = flput;
	  break;
	case 13:
	  A_plus[i] = flput;
	  break;
	case 14:
	  A_minus[i] = flput;
	  break;
	case 15:
	  tau_plus[i] = flput;
	  break;
	case 16:
	  tau_minus[i] = flput;
	  break;
	case 17:
	  plastic[i] = bput;
	  break;
	case 18:
	  max_weight[i] = flput;
	  break;
	case 19:
	  record[i] = bput;
	  break;
	}
	last = next + 1;
      }
    }
    cout << "calculate coefficients\n";
    //calculate class-level dynamics coefficients
    Cinv = new float[numClasses];
    vrPlusVt = new float[numClasses];
    kVrVt = new float[numClasses];
    ab = new float[numClasses];
    abVr = new float[numClasses];
    LTPdecay = new float[numClasses];
    LTDdecay = new float[numClasses];

    for (i = 0; i < numClasses; i++){
      Cinv[i] = 1/C[i];
      vrPlusVt[i] = vr[i] + vt[i];
      kVrVt[i] = kdyn[i] * vr[i] * vt[i];
      ab[i] = a[i] * b[i];
      abVr[i] = a[i] * b[i] * vr[i];
      LTPdecay[i] = 1.0 - (1.0/tau_plus[i]);
      LTDdecay[i] = 1.0 - (1.0/tau_minus[i]);
    }
    cout << "parse neuron params\n";
    //parse neuron-level params (i.e. dynamics class ID)
    unitClass = new int[N];
    for (i = 0; i < N; i++){
      getline(infile, line);
      unitClass[i] = stoi(line.substr(0,string::npos));
    }
    post = new int[N*M];
    s = new float[N*M];
    sd = new float[N*M];
    delays_length = new short[N*D];
    delays = new short[N*D*M];
    for (i = 0; i < N * D; i++){
      delays_length[i] = 0;
      for (j = 0; j < M; j++){
	delays[i * N * D + j] = -1;
      }
    }

    cout << "parse synapse params\n";
    //parse synapse-level params
    short d;
    for (i = 0; i < N; i++){
      //cout << i << "\n";
      for (j=0; j < M; j++){
	getline(infile, line, ';');
	for (k = 0; k < 4; k++){
	  next = line.find(",", last);
	  switch(k){
	  case 0:
	    post[i*M+j] = stoi(line.substr(last, next - last));
	    break;
	  case 1:
	    s[i*M+j] = stof(line.substr(last, next - last));
	    break;
	  case 2:
	    sd[i*M+j] = stof(line.substr(last, next - last));
	    break;
	  case 3:
	    d = (short)stoi(line.substr(last, next - last));
	    if (d <= D) {
	      delays[i * D * M + (d - 1) * M
		     + delays_length[i * D + (d - 1)]] = j;
	      delays_length[i * D + (d - 1)]++;
	    } else {
	      cout << d << "\n";
	      exit(2);
	    }
	    break;
	  }
	  last = next + 1;
	}

      }
      getline(infile, line); //throw out the rest of the line
    }

    N_pre = new int[N];
    I_pre = new int[N * 3 * M];
    D_pre = new int[N * 3 * M];
    s_pre = new float*[N * 3 * M];
    sd_pre = new float*[N * 3 * M];
    cout << "set up pointers\n";
    for (i = 0; i < N; i++) {
      N_pre[i] = 0;
      // Note: synapses are not plastic from inh neurons in default net;
      // in general, this is not the case, so we examine all presynaptic
      // neurons here (use N instead of only count of exc neurons).
      for (j = 0; j < N; j++) { 
	for (k = 0; k < M; k++) {
	  if (post[j * M + k] == i) {		// find all presynaptic neurons
	    I_pre[i * 3 * M + N_pre[i]] = j; // add this neuron to the list
	    for (dd = 0; dd < D; dd++)	// find the delay
	      for (jj = 0; jj < delays_length[j * D + dd]; jj++)
		if (post[j * M + delays[j * D * M + dd * M + jj]]
		    == i)
		  D_pre[i * 3 * M + N_pre[i]] = dd;
	    s_pre[i * 3 * M + N_pre[i]] = &s[j * M + k];// pointer to the synaptic weight
	    sd_pre[i * 3 * M + N_pre[i]++] = &sd[j * M + k];// pointer to the derivative
	  }
	}
      }
    }
  }
  cout << "done\n";
}
/*--------------------------------------------------------------
  void	SpikingNetwork::polychronous(int nnum){
  int	i,j, t, p, k;
  int npre[W];
  int dd;
  int	t_last, timing;
  int	Dmax, L_max; 
  int	used[W], discard;
  int			N_polychronous;
  int			N_postspikes[polylenmax], I_postspikes[polylenmax][N], J_postspikes[polylenmax][N], D_postspikes[polylenmax][N], L_postspikes[polylenmax][N];
  double		C_postspikes[polylenmax][N];
  int			N_links, links[2*W*polylenmax][4];
  int			group[polylenmax], t_fired[polylenmax], layer[polylenmax];
  int			gr3[W], tf3[W];
  int			I_my_pre[3*M], D_my_pre[3*M], N_my_pre;
  int			N_fired;
  FILE		*fpoly;


  double		C_rel = 0.95*C_max;
 
  double v[N],u[N],I[N];
 
 
  N_my_pre = 0;
  for (i=0;i<N_pre[nnum];i++){
  if (*s_pre[nnum][i] > C_rel) {
  I_my_pre[N_my_pre]=I_pre[nnum][i];
  D_my_pre[N_my_pre]=D_pre[nnum][i];
  N_my_pre++;
  }
  }
  if (N_my_pre<W){
  return;
  }
  for (i=0;i<W;i++){
  npre[i]=i;
  }
 
  while (0==0){
  Dmax=0;
  for (i=0;i<W;i++){
  if (Dmax < D_my_pre[npre[i]]){
  Dmax=D_my_pre[npre[i]];
  }
  }
  for (i=0;i<W;i++)	{
  group[i]=I_my_pre[npre[i]];
  t_fired[i]= Dmax-D_my_pre[npre[i]];
  layer[i]=1;
 
  for (dd=0; dd<D; dd++){		 
  for (j=0; j<delays_length[group[i]][dd]; j++) {
  p = post[group[i]][delays[group[i]][dd][j]];
  if ((s[group[i]][delays[group[i]][dd][j]] > C_rel) & (dd>=D_my_pre[npre[i]])) {
  timing = t_fired[i]+dd+1;
  J_postspikes[timing][N_postspikes[timing]]=group[i];				// presynaptic
  D_postspikes[timing][N_postspikes[timing]]=dd;					// delay
  C_postspikes[timing][N_postspikes[timing]]=s[group[i]][delays[group[i]][dd][j]];	// syn weight
  I_postspikes[timing][N_postspikes[timing]++]=p;					// index of post target	
  }
  }
  }
  }

  for (i=0;i<N;i++){
  v[i]=-70;
  u[i]=0.2*v[i];
  I[i]=0;
  }

  N_links = 0;
  N_fired=W;
  t_last = D+D+latency+1;
  t=-1;
  while ((++t<t_last) & (N_fired < polylenmax)){
  for (p=0;p<N_postspikes[t];p++){ 
  I[I_postspikes[t][p]]+=C_postspikes[t][p]; 
  }
  for (i=0;i<N;i++) {
  v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]);
  v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]);
  u[i]+=a[i]*(0.2*v[i]-u[i]);
  I[i]=0;
  }

  for (i=0;i<N;i++){ 
  if (v[i]>=30) {
  v[i] = -65;
  u[i]+=d[i];

  if (N_fired < polylenmax){
  t_fired[N_fired]= t;
  group[N_fired++]=i;
  for (dd=0; dd<D; dd++){
  for (j=0; j<delays_length[i][dd]; j++){
  if ((s[i][delays[i][dd][j]] > C_rel) | (i>=Ne)) {
  timing = t+dd+1;
  J_postspikes[timing][N_postspikes[timing]]=i;				// presynaptic
  D_postspikes[timing][N_postspikes[timing]]=dd;				// delay
  //L_postspikes[timing][N_postspikes[timing]]=NL+1;			// layer
  C_postspikes[timing][N_postspikes[timing]]=s[i][delays[i][dd][j]];	   // syn weight
  I_postspikes[timing][N_postspikes[timing]++]=post[i][delays[i][dd][j]];// index of post target	
  }
  }
  }
  if (t_last < timing+1) {
  t_last = timing+1;
  if (t_last > polylenmax-D-1){
  t_last = polylenmax-D-1;
  }
  }
  }
  }
  }
  }
 
  if (N_fired>2*W){
  N_links=0;
  L_max=0;
  for (i=W;i<N_fired;i++){
  layer[i]=0;
  for (p=t_fired[i]; (p>t_fired[i]-latency) & (p>=0); p--){
  for (j=0;j<N_postspikes[p];j++){
  if ((I_postspikes[p][j]==group[i]) & (J_postspikes[p][j]<Ne)) {
  for (k=0;k<i;k++){
  if ((group[k]==J_postspikes[p][j]) & (layer[k]+1>layer[i])){
  layer[i]=layer[k]+1;
  }
  }
  //{   
  links[N_links][0]=J_postspikes[p][j];
  links[N_links][1]=I_postspikes[p][j];
  links[N_links][2]=D_postspikes[p][j];
  links[N_links++][3]=layer[i];
  if (L_max < layer[i]){
  L_max = layer[i];
  }
  //}
  }
  }
  }
  }
 
  discard = 0;
  for (i=0;i<W;i++) {
  used[i]=0;
  for (j=0;j<N_links;j++){
  if ((links[j][0] == group[i]) & (links[j][1] < Ne)){
  used[i]++;
  }
  }
  if (used[i] == 1) {
  discard = 1;
  }
  }

  //if ((discard == 0) & (t_fired[N_fired-1] > min_group_time) )  // (L_max >= min_group_path))
  if ((discard == 0) & (L_max >= min_group_path)) {

  for (i=0;i<W;i++) {
  gr3[i]=group[i];
  tf3[i]=t_fired[i];
  } //???

  N_polychronous++;
  cout << "\ni= " << nnum
  << ", N_polychronous= " << N_polychronous
  << ", N_fired = " << N_fired
  << ", L_max = " << L_max
  << ", T=" << t_fired[N_fired-1];
 
  fprintf(fpoly, " %d  %d,       ", N_fired, L_max);
  for (i=0; i<N_fired; i++){
  fprintf(fpoly, " %d %d, ", group[i], t_fired[i]);
  }
  fprintf(fpoly, "        ");
  for (j=0;j<N_links;j++){
  fprintf(fpoly, " %d %d %d %d,  ", links[j][0], links[j][1], links[j][2], links[j][3]);
  }
  fprintf(fpoly, "\n");
  }
  }

  for (dd=Dmax;dd<t_last;dd++) {
  N_postspikes[dd]=0;
  }
  if (t_last == polylenmax-D) {
  for (dd=t_last;dd<polylenmax;dd++) {
  N_postspikes[dd]=0;
  }
  }
  i=1;
  while (++npre[W-i] > N_my_pre-i){
  if (++i > W){
  return;
  }
  }
  while (i>1) {
  npre[W-i+1]=npre[W-i]+1; i--;
  }
  }
  }  

 
  //--------------------------------------------------------------
  void	SpikingNetwork::all_polychronous()
  {
  int	i;
  N_polychronous=0;
  fpoly = fopen(".//polyall.dat","w"); //necessary??
  for (i=0;i<polylenmax;i++) N_postspikes[i]=0;
 
  for (i=0;i<Ne;i++) polychronous(i);
 
  cout << "\nN_polychronous=" << N_polychronous << "\n";
  fclose(fpoly); 
  }
*/

// TODO TODO TODO:
// This has become a mess. Need to factor out input and output, set engine object params,
// move engine out of network class, etc.
void SpikingNetwork::simulate(int maxSecs, int trainSecs, int testSecs,
			      string inFileName, float scale, bool kaldiMode,
			      int numFeats, int stepSize, string outFileName, int frequency) {
  cout << "simulate\n";
  short step = 10; //default
  //float scale = 1.0;
  //float labelScale = 1.0;
  //float shift = 0.66;
  //ofstream labelData;
  //int inFeat = 0;
  double updateMs = 1000.0 / frequency;
  bool done = false;
  bool preDone = false;
  bool test = false;
  ulong i, j, k, sec, t;
  float I[N];
  bool fileInput = false;
  ifstream inputData;
  string inputLine;
  string currentLine;
  int feat = 0;
  ofstream outputFile;
  ofstream indexFile;
  ulong outByteCount = 0;
  sec = 0;
  queue<string> headers;
  queue<int> headerTimes;
  queue<int> tailTimes;

  int N_recorded_units = 0;
  for (i = 0; i < numClasses; i++){
    N_recorded_units += record[i] ? count[i] : 0;
  }
  int N_recorded_firings = 0;
  int N_firings = 1;				// the number of fired neurons 
  const int N_firings_max = 150 * N * (frequency/1000);// upper limit on the number of fired neurons per sec
  int firings[N_firings_max][2];              // indices and timings of spikes

  // spike timings
  firings[0][0] = -D;	// put a dummy spike at -D for simulation efficiency 
  firings[0][1] = 0;	// index of the dummy spike  

  if (inFileName != "") {
    fileInput = true;
    inputData.open(inFileName);
    //labelData.open("labels.txt");
    if (inputData.is_open()) {
      if (!kaldiMode){
	getline(inputData, inputLine);
	//TODO: make this suck less.
	string tok;
	size_t pos = 0;
	pos = inputLine.find(' ');
	tok = inputLine.substr(0, pos);
	step = stoi(tok);
	inputLine.erase(0, pos + 1);
	pos = inputLine.find(' ');
	tok = inputLine.substr(0, pos);
	feat = stoi(tok);
	// cout << feat;
	// removing notion of "labels"
	//tok = inputLine.substr(pos, inputLine.length() - 2);
	//inFeat = stoi(tok);
	//cout << inFeat;
      } else {
	step = stepSize;
	feat = numFeats;
	//inFeat = numFeats;
      }
      //load first input line
      getline(inputData, currentLine);
      //pre-load next line (almost certainly not getting a benefit from this)
      getline(inputData, inputLine); 
      if (kaldiMode){
	size_t next = currentLine.find_first_not_of(" \t\r\n");
	if (!isdigit(currentLine[next])){
	  //queue line to output to ark
	  headers.push(currentLine);
	  headerTimes.push(t);
	  currentLine = inputLine;
	  getline(inputData, inputLine);
	}
      }
    } else {
      cout << "Could not open file." << endl;
    }
  }
  // removing notion of "labels"

  //int lastLabel = feat;
  //const int numLabels = feat - inFeat;
  //int labelSpikes[numLabels];

  //for (i = 0; i < numLabels; i++) {
  //  labelSpikes[i] = 0;
  //}

  LTP = new float[N * (frequency + 1 + D)];
  LTD = new float[N];
  v = new float[N];
  u = new float[N];
  for (i = 0; i < N; i++) {
    for (j = 0; j < 1 + D; j++) {
      LTP[i * (frequency + 1 + D) + j] = 0.0;
    }
  }
  for (i = 0; i < N; i++) {
    LTD[i] = 0.0;
  }
  for (i = 0; i < N; i++) {
    v[i] = -65.0; // initial values for v
  }
  for (i = 0; i < N; i++) {
    u[i] = 0.2 * v[i];	// initial values for u
  }
  
  short framesLeft = step - 1;
  for (i = 0; i < N; i++) {
    I[i] = 0.0;	// reset the input
  }
  char currentPath[FILENAME_MAX];
  if (!getcwd(currentPath, sizeof(currentPath))){
    cerr << "Error: getcwd()";
    return;
  }
  //make filename absolute
  if (outFileName[0] != '/'){
    string cwd(currentPath);
    outFileName = cwd + "/" + outFileName;
  }
  outputFile.open(outFileName);
  if (kaldiMode){
    size_t pos = outFileName.find_last_of(".");
    if (pos != string::npos) {
      indexFile.open(outFileName.substr(0,pos)+".scp");
    } else {
      indexFile.open(outFileName + ".scp");
    }
  }
  ulong testCounter =  0;
  //TODO: runoff
  int runoff = 1;
  bool endUtt = false;
  while (!done)		// different ways to be done
    {
      if (trainSecs > 0 && sec % (trainSecs + testSecs) == 0) {
	test = false;
      } else if (sec % (trainSecs + testSecs) - trainSecs == 0) {
	test = true;
	testCounter = 0;
	//all_polychronous();
      }
      t = 0;
      while (t < frequency && !done)				// simulation of 1 sec
	{
	  if (!fileInput) {
	    for (i = 0; i < N; i++) {
	      I[i] = 0.0;	// reset the input
	    }
	    for (k = 0; k < N / 1000; k++) {
	      I[getrandom(N)] = 20.0 * scale; // random thalamic input
	    }
	  } else { // file input
	    int ii;
	    for (ii = feat; ii < N; ii++) {
	      I[ii] = 0.0;
	    }
	    //parse input
	    size_t next = currentLine.find_first_not_of(" \t\r\n");
	    size_t last = next;
	    for (ii = 0; ii < feat; ii++) {
	      next = currentLine.find(" ", last);
	      // removing notion of "labels"
	      //if (ii < inFeat) { //input neurons
		I[ii] = stof(currentLine.substr(last, next - last))
		  * scale;
	      //  } else { //label neurons
 	      // 	if (test) {
	      // 	  int q = 0;
	      // 	  I[ii] = 0.0;
	      // 	  //check for active label
	      // 	  //if same as last time, leave it, otherwise switch
	      // 	  if (stof(currentLine.substr(last, next - last))
	      // 	      > 0.5 && ii != lastLabel) {
	      // 	    labelData << "sec= " << sec << ", Label="
	      // 		      << lastLabel - inFeat << ", [ ";
	      // 	    for (q = 0; q < numLabels; q++) {
	      // 	      labelData << labelSpikes[q] << " ";
	      // 	    }
	      // 	    labelData << "]" << endl;
	      // 	    for (q = 0; q < numLabels; q++) {
	      // 	      labelSpikes[q] = 0;
	      // 	    }
	      // 	    lastLabel = ii;
	      // 	  }
	      // 	} else {
	      // 	  I[ii] = (stof(currentLine.substr(last, next - last))
	      // 		   - shift) * labelScale;
	      // 	}
	      // }
	      last = next + 1;
	    }
	    if (framesLeft == 0) {
	      //load next line (or stop)
	      currentLine = inputLine;
	      getline(inputData, inputLine);
	      if (kaldiMode){
		size_t next = currentLine.find_first_not_of(" \t\r\n");
		if (currentLine != "" && currentLine[currentLine.length()-1]=='['){
		  //queue line to output to ark
		  //cout << "header at t = " << t << "\n";
		  headers.push(currentLine);
		  headerTimes.push(t);
		  currentLine = inputLine;
		  getline(inputData, inputLine);
		}
		if (endUtt){ // not "done" if we get here
		  endUtt = false;
		  runoff = 1;
		  //queue ']' for output
		  //cout << "tail at t = " << t << "\n";
		  tailTimes.push(t);
		}
		if (currentLine[currentLine.length()-1] == ']'){
		  endUtt = true;
		}
	      }
	      if (preDone) {
		done = true;
	      }
	      if (inputData.eof() && !preDone){
		preDone = true;
	      }
	      //reset framesLeft
	      //if (!done){
		framesLeft = step - 1;
		//}
	    } else {
	      framesLeft--;
	    }
	  }
	  //find all firings for the current time step, do necessary updates.
	  for (i = 0; i < N; i++) {
	    if (v[i] >= peak[unitClass[i]])    // did it fire?
	      {
		if (record[unitClass[i]]) {N_recorded_firings++;}
		v[i] = c[unitClass[i]];	// voltage reset
		u[i] += d[unitClass[i]];	// recovery variable reset
		LTP[i * (frequency + 1 + D) + (t + D)] = A_plus[unitClass[i]];
		LTD[i] = A_minus[unitClass[i]];
		for (j = 0; j < N_pre[i]; j++) {
		  // this spike was after pre-synaptic spikes;
		  // add to the weight of the synapse between this neuron and its
		  // pre-synaptic neuron according to the LTP value for the presynaptic
		  // neuron at the delay time in the past (this is why we need to store
		  // values for all times).

		  // generalization handled by setting LTP curve parameters on class-basis
		  // (would a switch save computation?)
		  *sd_pre[i * 3 * M + j] += LTP[I_pre[i * 3 * M + j]
						* (frequency + 1 + D)
						+ (t + D - D_pre[i * 3 * M + j] - 1)];
		}
		firings[N_firings][0] = t;
		firings[N_firings++][1] = i;
		if (N_firings == N_firings_max) {
		  cout << "Too many spikes at t=" << t
		       << " (ignoring all)" << endl;
		  N_firings = 1;
		}
		// removing notion of "labels"
		// if (test && inFeat <= i && i < feat) {
		//   labelSpikes[i - inFeat]++;
		// }
	      }
	  }
	  k = N_firings;
	  //look for previous firings at every possible delay value
	  //(we want to find previous firings that affect the current update)
	  while (t - firings[--k][0] < D) {
	    //for the neuron that fired, for each efferent with
	    //the delay value == now - then
	    for (j = 0;
		 j < delays_length[firings[k][1] * D + (t - firings[k][0])];
		 j++) {
	      // i is the index of the neuron that gets delayed input from neuron at firings[k][1] now
	      i = post[firings[k][1] * M
		       + delays[firings[k][1] * D * M
				+ (t - firings[k][0]) * M + j]];
	      I[i] += s[firings[k][1] * M
			+ delays[firings[k][1] * D * M
				 + (t - firings[k][0]) * M + j]];
	      // this spike is after postsynaptic spikes (past spikes at neuron i)	
	      // LTD[i] spikes when neuron i spikes, then decays.
	      // depression is only affected by most recent spike
	      // non-plastic synapses handled by parameters of LTD curve.
	      sd[firings[k][1] * M
		 + delays[firings[k][1] * D * M
			  + (t - firings[k][0]) * M + j]] -=
		LTD[i];
	    }
	  }

	  for (i = 0; i < N; i++) {
	    // for numerical stability time step is half the update frequency
	    v[i] += 0.5 * updateMs * Cinv[unitClass[i]] * (kdyn[unitClass[i]] * (v[i] - vrPlusVt[unitClass[i]])
						* v[i] + kVrVt[unitClass[i]] - u[i]) + I[i];
	    v[i] += 0.5 * updateMs * Cinv[unitClass[i]] * (kdyn[unitClass[i]] * (v[i] - vrPlusVt[unitClass[i]])
						* v[i] + kVrVt[unitClass[i]] - u[i]) + I[i];
	    u[i] += a[unitClass[i]] * ((v[i] < caInact[unitClass[i]] ? bhyp[unitClass[i]] : b[unitClass[i]])
					        * (v[i] - vr[unitClass[i]]) - u[i]);
	    u[i] = min(umax[unitClass[i]],u[i]);
	    LTP[i * (frequency + 1 + D) + (t + D + 1)] = LTPdecay[unitClass[i]] * updateMs
	      * LTP[i * (frequency + 1 + D) + (t + D)];
	    LTD[i] *= LTDdecay[unitClass[i]] * updateMs;
	  }
	  t++;
	}
      //cout << "sec=" << sec << ", firing rate=" << float(N_firings) / N << "\n";
      //dense output in kaldi-mode, sparse (octave-compatible) otherwise.
      if (test){
	if (kaldiMode) {
	  //note: for this to work, spikes at same time must have been appended in index order.
	  //also, this may be unnecessarily complicated, but I was trying to avoid
	  //evaluating conditionals for every neuron at every millisecond.
	  j = 0;
	  k = 0;
	  ulong l = 0;
	  ulong base = 0;
	  size_t pos = 0;
	  string key = "";
	  ulong offset = 0;
	  if (!tailTimes.empty() && tailTimes.front() == j) {
	    outputFile << "]\n";
	    outByteCount += 2;
	    tailTimes.pop();
	  }
	  if (!headerTimes.empty() && headerTimes.front() == j){
	    string uttHead = headers.front();
	    outputFile << uttHead << "\n  ";
	    outByteCount += (uttHead.length() + 3);
	    pos = uttHead.find_first_of(" \t");
	    key = uttHead.substr(0, pos);
	    offset = outByteCount - 4;
	    indexFile << key + " " + outFileName + ":" + to_string(offset) + "\n";
	    headers.pop();
	    headerTimes.pop();
	  }
	  for (i = 1; i < N_firings; i++){
	    if (firings[i][0] >= 0){  //rolled-over spikes have negative timings
	      if (record[unitClass[firings[i][1]]]){ //only write if it's a recorded neuron.
		//write everything that (didn't) happen in the interim
		while (j < firings[i][0]){ //catch up to current step
		  while (k < numClasses){
		    if (record[k]){
		      while (l < count[k]){
			outputFile << "0 ";
			l++;
			outByteCount += 2;
		      }
		    }
		    k++;
		    base = base + count[k-1];
		    l = 0;
		  }
		  //find interrupts
		  if (!tailTimes.empty() && tailTimes.front() == j) {
		    outputFile << "]\n";
		    outByteCount += 2;
		    tailTimes.pop();
		  } else {
		    outputFile << "\n  ";
		    outByteCount += 3;
		  }
		  if (!headerTimes.empty() && headerTimes.front() == j){
		    string uttHead = headers.front();
		    outputFile << uttHead << "\n  ";
		    outByteCount += (uttHead.length() + 3);
		    pos = uttHead.find_first_of(" \t");
		    key = uttHead.substr(0, pos);
		    offset = outByteCount - 4;
		    indexFile << key + " " + outFileName + ":" + to_string(offset) + "\n";
		    headers.pop();
		    headerTimes.pop();
		  }
		  j++;
		  k = 0;
		  base = 0;
		}
		//now we're in the right millisecond, get into the right class.
		while (k < unitClass[firings[i][1]]){
		  if (record[k]){
		    while (l < count[k]){
		      outputFile << "0 ";
		      outByteCount += 2;
		      l++;
		    }
		  }
		  k++;
		  base = base + count[k-1];
		  l = 0;
		}
		//now we're in the right class, get to the right neuron
		while (l < firings[i][1] - base){ 
		  outputFile << "0 ";
		  outByteCount += 2;
		  l++;
		}
		//output the spike
		outputFile << "1 ";
		outByteCount += 2;
		l++;
	      }
	    }
	  }
	  //done with last spike, pad out to the end of the second
	  while (j < t){ //go until the last step that was simulated
	    while (k < numClasses){
	      if (record[k]){
		while (l < count[k]){
		  outputFile << "0 ";
		  outByteCount += 2;
		  l++;
		}
	      }
	      k++;
	      base = base + count[k-1];
	      l = 0;
	    }
	    //find interrupts
	    if (!tailTimes.empty() && tailTimes.front() == j) {
	      outputFile << "]\n";
	      outByteCount += 2;
	      tailTimes.pop();
	    } else {
	      outputFile << "\n  ";
	      outByteCount += 3;
	    }
	    if (!headerTimes.empty() && headerTimes.front() == j){
	      string uttHead = headers.front();
	      outputFile << uttHead << "\n  ";
	      outByteCount += (uttHead.length() + 3);
	      pos = uttHead.find_first_of(" \t");
	      key = uttHead.substr(0, pos);
	      offset = outByteCount - 4;
	      indexFile << key + " " + outFileName + ":" + to_string(offset) + "\n";
	      headers.pop();
	      headerTimes.pop();
	    }
	    j++;
	    k = 0;
	    base = 0;
	  }
	} else {
	  for (i = 1; i < N_firings; i++) {
	    if (firings[i][0] >= 0) {
	      outputFile << firings[i][0] + (testCounter * frequency) << " " << firings[i][1] << "\n";
	    }
	  }
	}
      }
      // prepare for the next sec

      // roll over LTP data to prefix of next second
      for (i = 0; i < N; i++) {	
	for (j = 0; j < D + 1; j++) {
	  LTP[i * (frequency + 1 + D) + j] = LTP[i * (frequency + 1 + D) + (frequency + j)];
	}
      }
      k = N_firings - 1;
      //find first firing within delay period
      while (frequency - firings[k][0] < D) {
	k--;
      }
      // roll over firings within delay period
      for (i = 1; i < N_firings - k; i++) {
	firings[i][0] = firings[k + i][0] - frequency;
	firings[i][1] = firings[k + i][1];
      }
      N_firings = N_firings - k;

      for (i = 0; i < N; i++) {
	if (plastic[unitClass[i]]){// modify only plastic connections
	  for (j = 0; j < M; j++) {
	    s[i * M + j] += 0.01 + sd[i * M + j]; //only place for weight modification
	    sd[i * M + j] *= 0.9; // ??
	    if (abs(s[i * M + j]) > abs(max_weight[unitClass[i]])){
	      s[i * M + j] -= s[i * M + j] - max_weight[unitClass[i]];
	    }
	    if (max_weight[unitClass[i]] > 0 && s[i * M + j] < 0) {
	      s[i * M + j] = 0.0;
	    }
	    if (max_weight[unitClass[i]] < 0 && s[i * M + j] > 0) {
	      s[i * M + j] = 0.0;
	    }
	  }
	}
      }
      sec++;
      testCounter = test ? testCounter + 1 : test;
      if (runoff > 0) {
	runoff--;
      }
      if (sec == maxSecs) {
	done = true;
      }
    }
  cout << "Recorded spikes: " << N_recorded_firings << "\n";
  cout << "Number of recorded units: " << N_recorded_units << "\n";
  cout << "Seconds simulated: " << (double)sec + (t / frequency) << "\n";
  cout << "Average recorded firing rate: " << (double)N_recorded_firings / (sec + (t / frequency)) / N_recorded_units << "\n";
  outputFile.close();
  if (inputData.is_open()) {
    inputData.close();
  }
  if (indexFile.is_open()) {
    indexFile.close();
  }

}

int main(int argc, char *argv[]) {
  int maxSecs = 60;
  int trainSecs = 60;
  int testSecs = 0;
  //int Nexc, Ninh;
  
  try {
    CmdLine cmd("Run a spiking network simulation under supplied parameters.",' ', "0.1");
    ValueArg < string
	       > inFile("i", "input", "name of file containing input values",
			false, "", "string");
    ValueArg < string
	       > netFile("N", "network", "name of file containing network configuration",
			false, "", "string");
    ValueArg<int> maxTime("M", "max",
			  "maximum number of seconds to simulate", false, 60, "integer");
    ValueArg<int> trainTime("t", "train",
			    "number of seconds in the training interval", false, 60,
			    "integer");
    ValueArg<int> testTime("x", "test",
			   "number of seconds in the testing interval", false, 0,
			   "integer");
    ValueArg<float> scaleArg("c", "scale",
			   "coefficient by which to scale input values", false, 1.0,
			   "float");
    SwitchArg kaldiArg("k","kaldi-mode",
		       "enable reading and writing of kaldi-compatible files;\n\
Requires --num-feats and --step to be specified");
    ValueArg<int> numFeatsArg("n", "num-feats",
			      "number of features on each line of the input file",
			      false, 0, "integer");
    ValueArg<int> stepArg("s", "step", "number of time steps (defined by --frequency) per input frame", false, 0, "integer");
    ValueArg <string> outFileArg("o", "output",
				 "name of file to which to write spike timings",
				 false, "spikes.dat", "string");
    ValueArg<int> frequencyArg("f", "frequency", "input sample rate in Hz", false, 1000, "integer");
    
    cmd.add(inFile);
    cmd.add(maxTime);
    cmd.add(trainTime);
    cmd.add(testTime);
    cmd.add(netFile);
    cmd.add(scaleArg);
    cmd.add(kaldiArg);
    cmd.add(numFeatsArg);
    cmd.add(stepArg);
    cmd.add(outFileArg);
    cmd.add(frequencyArg);
    cmd.parse(argc, argv);
    string fileHandle = inFile.getValue();
    string netFilename = netFile.getValue();
    float scale = scaleArg.getValue();
    bool kaldiMode = kaldiArg.getValue();
    int numFeats = numFeatsArg.getValue();
    int stepSize = stepArg.getValue();
    string outFile = outFileArg.getValue();
    int frequency = frequencyArg.getValue();
    
    if (kaldiMode && (numFeats == 0 || stepSize == 0)){
      cerr << "Kaldi mode requires specification of num-feats and step";
      return 1;
    }
    maxSecs = maxTime.getValue();
    trainSecs = trainTime.getValue();
    testSecs = testTime.getValue();
    SpikingNetwork* net;
    // assign connections, weights, etc.
    if (netFilename == ""){
      net = new SpikingNetwork();
    } else {
      net = new SpikingNetwork(netFilename);
    }
    // tests of constructors and serializer
    //net->saveTo("network-test.dat");
    //SpikingNetwork* net2 = new SpikingNetwork("network.dat");
    //net2->saveTo("network2.dat");

    // ridiculous
    net->simulate(maxSecs, trainSecs, testSecs, fileHandle, scale, kaldiMode, numFeats, stepSize, outFile, frequency);
  } catch (ArgException &e) {
    //do stuff
    cerr << e.what();
    return 1;
  }

  return 0;
}
