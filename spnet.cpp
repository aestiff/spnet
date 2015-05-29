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
#include <tclap/CmdLine.h>

#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1
using namespace std;
using namespace TCLAP;

const   int		Ne = 800;		// excitatory neurons			
const   int		Ni = 200;		// inhibitory neurons				 
const	int		N  = Ne+Ni;		// total number of neurons	
const	int		M  = 100;		// the number of synapses per neuron 
const	int		D  = 20;		// maximal axonal conduction delay
float	sm = 10.0;		                // maximal synaptic strength		
int	post[N][M];				// indeces of postsynaptic neurons
float	s[N][M], sd[N][M];		        // matrix of synaptic weights and their derivatives
short	delays_length[N][D];	                // distribution of delays
short	delays[N][D][M];		        // arrangement of delays   
int	N_pre[N], I_pre[N][3*M], D_pre[N][3*M];	// presynaptic information
float	*s_pre[N][3*M], *sd_pre[N][3*M];	// presynaptic weights
float	LTP[N][1001+D], LTD[N];	                // STDP functions 
float	a[N], d[N];				// neuronal dynamics parameters
float	v[N], u[N];				// activity variables
int	N_firings;				// the number of fired neurons 
const   int         N_firings_max=100*N;	// upper limit on the number of fired neurons per sec
int	firings[N_firings_max][2];              // indeces and timings of spikes
double	C_max=10;		
const	int	W=3;	                        // initial width of polychronous groups
int     min_group_path = 7;		        // minimal length of a group
int	min_group_time = 40;	                // minimal duration of a group (ms)

void initialize()
{
  //  Ne = Nexc;
  //  Ni = Ninh;
  int i,j,k,jj,dd, exists, r;
  for (i=0;i<Ne;i++){
    a[i]=0.02;// RS type
  }
  for (i=Ne;i<N;i++){
    a[i]=0.1; // FS type
  }
  for (i=0;i<Ne;i++){
    d[i]=8.0; // RS type
  }
  for (i=Ne;i<N;i++){
    d[i]=2.0; // FS type
  }
  for (i=0;i<N;i++){
    for (j=0;j<M;j++) 
      {
	do{
	  exists = 0;		// avoid multiple synapses
	  if (i<Ne) r = getrandom(N);
	  else	  r = getrandom(Ne);// inh -> exc only
	  if (r==i) exists=1;				      	// no self-synapses 
	  for (k=0;k<j;k++) if (post[i][k]==r) exists = 1;	// synapse already exists  
	}while (exists == 1);
	post[i][j]=r;
      }
  }
  for (i=0;i<Ne;i++){
    for (j=0;j<M;j++){
      s[i][j]=6.0; // initial exc. synaptic weights
    }
  }
  for (i=Ne;i<N;i++){
    for (j=0;j<M;j++){
      s[i][j]=-5.0; // inhibitory synaptic weights
    }
  }
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      sd[i][j]=0.0; // synaptic derivatives 
    }
  }
  for (i=0;i<N;i++)
    {
      short ind=0;
      if (i<Ne)
	{
	  for (j=0;j<D;j++) 
	    {	delays_length[i][j]=M/D;	// uniform distribution of exc. synaptic delays
	      for (k=0;k<delays_length[i][j];k++){
		delays[i][j][k]=ind++;
	      }
	    }
	}
      else
	{
	  for (j=0;j<D;j++){
	    delays_length[i][j]=0;
	  }
	  delays_length[i][0]=M;			// all inhibitory delays are 1 ms
	  for (k=0;k<delays_length[i][0];k++){
	    delays[i][0][k]=ind++;
	  }
	}
    }
  
  for (i=0;i<N;i++)
    {
      N_pre[i]=0;
      for (j=0;j<Ne;j++){
	for (k=0;k<M;k++){
	  if (post[j][k] == i){		// find all presynaptic neurons 
	    I_pre[i][N_pre[i]]=j;	// add this neuron to the list
	    for (dd=0;dd<D;dd++)	// find the delay
	      for (jj=0;jj<delays_length[j][dd];jj++)
		if (post[j][delays[j][dd][jj]]==i) D_pre[i][N_pre[i]]=dd;
	    s_pre[i][N_pre[i]]=&s[j][k];	// pointer to the synaptic weight	
	    sd_pre[i][N_pre[i]++]=&sd[j][k];// pointer to the derivative
	  }
	}
      }
    }
  
  for (i=0;i<N;i++){
    for (j=0;j<1+D;j++){
      LTP[i][j]=0.0;
    }
  }
  for (i=0;i<N;i++){
    LTD[i]=0.0;
  }
  for (i=0;i<N;i++){
    v[i]=-65.0; // initial values for v
  }
  for (i=0;i<N;i++){
    u[i]=0.2*v[i];	// initial values for u
  }
  N_firings=1;		// spike timings
  firings[0][0]=-D;	// put a dummy spike at -D for simulation efficiency 
  firings[0][1]=0;	// index of the dummy spike  
}

const	int	latency = D; // maximum latency 
//--------------------------------------------------------------
int			N_polychronous;


double		C_rel = 0.95*C_max;
const int	polylenmax = N;
int			N_postspikes[polylenmax], I_postspikes[polylenmax][N], J_postspikes[polylenmax][N], D_postspikes[polylenmax][N], L_postspikes[polylenmax][N];
double		C_postspikes[polylenmax][N];
int			N_links, links[2*W*polylenmax][4];
int			group[polylenmax], t_fired[polylenmax], layer[polylenmax];
int			gr3[W], tf3[W];
int			I_my_pre[3*M], D_my_pre[3*M], N_my_pre;
int			N_fired;


FILE		*fpoly;


//--------------------------------------------------------------
void	polychronous(int nnum){
  int	i,j, t, p, k;
  int npre[W];
  int dd;
  int	t_last, timing;
  int	Dmax, L_max; 
  int	used[W], discard;
  
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
	    
	    //fprintf(fpoly, " %d  %d,       ", N_fired, L_max);
	    //for (i=0; i<N_fired; i++)
	    //fprintf(fpoly, " %d %d, ", group[i], t_fired[i]);
	    //fprintf(fpoly, "        ");
	    //for (j=0;j<N_links;j++){
	    //   fprintf(fpoly, " %d %d %d %d,  ", links[j][0], links[j][1], links[j][2], links[j][3]);
	    //}
	    //fprintf(fpoly, "\n");
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
}  

 
//--------------------------------------------------------------
void	all_polychronous()
{
  int	i;
  N_polychronous=0;
  fpoly = fopen("..//polyall.dat","w"); //necessary??
  for (i=0;i<polylenmax;i++) N_postspikes[i]=0;
  
  for (i=0;i<Ne;i++) polychronous(i);
  
  cout << "\nN_polychronous=" << N_polychronous << "\n";
  fclose(fpoly);
}




int main(int argc, char *argv[]){
  int		i, j, k, sec, t;
  float	I[N];
  FILE	*fs;
  bool fileInput = false;
  ifstream inputData;
  ofstream labelData;
  short step = 10;
  int feat = 0;
  int inFeat = 0;
  int maxSecs = 60;
  int trainSecs = 60;
  int testSecs = 0;
  string inputLine;
  string currentLine;
  //int Nexc, Ninh;

  try{
    CmdLine cmd("Run a spiking network simulation under supplied parameters.",' ',"0.1");
    ValueArg<string> inFile("i","input","name of file containing input values",false,"","string");
    //ValueArg<int> excite("E","excite","number of excitatory neurons",false,800,"integer");
    //ValueArg<int> inhibit("I","inhibit","number of inhibitory neurons",false,200,"integer");
    ValueArg<int> maxTime("M","max","maximum number of seconds to simulate",false,60,"integer");
    ValueArg<int> trainTime("t","train","number of seconds in the training interval",false,60,"integer");
    ValueArg<int> testTime("x","test","number of seconds in the testing interval",false,0,"integer");
    cmd.add(inFile);
    cmd.add(maxTime);
    cmd.add(trainTime);
    cmd.add(testTime);
    //cmd.add(excite);
    //cmd.add(inhibit);
    cmd.parse(argc, argv);
    string fileHandle = inFile.getValue();
    if (fileHandle != ""){
      fileInput = true;
      inputData.open(fileHandle);
      labelData.open("labels.txt");
      if (inputData.is_open()){
	getline(inputData, inputLine);
	//TODO: make this suck less.
	string tok;
	size_t pos = 0;
	pos = inputLine.find(' ');
	tok = inputLine.substr(0,pos);
	step = stoi(tok);
	inputLine.erase(0,pos+1);
	pos = inputLine.find(' ');
	tok = inputLine.substr(0,pos);
	feat = stoi(tok);
	cout << feat;
	tok = inputLine.substr(pos, inputLine.length()-2);
	inFeat = stoi(tok);
	cout << inFeat;
	//load first input line
	getline(inputData, currentLine);
	getline(inputData, inputLine);
      } else {
	cout << "Could not open file." << endl;
      }
    }
    maxSecs = maxTime.getValue();
    trainSecs = trainTime.getValue();
    testSecs = testTime.getValue();
    //Nexc = excite.getValue();
    //Ninh = inhibit.getValue();
  } catch (ArgException &e){
    //do stuff
    cerr << e.what();
    return 1;
  }
    
  
  initialize();	// assign connections, weights, etc.  
  short framesLeft = step - 1;
  float scale = 0.50;
  float labelScale = 150.0;
  float shift = 0.66;
  bool done = false;
  sec = 0;
  bool preDone = false;
  bool test = false;
  int lastLabel = feat;
  const int numLabels = feat - inFeat;
  int labelSpikes[numLabels];

  
  for (i=0;i<numLabels;i++){
    labelSpikes[i] = 0;
  }
  for (i=0;i<N;i++){
    I[i] = 0.0;	// reset the input
  }
  while (!done)		// different ways to be done
    {
      if (sec % (trainSecs + testSecs) == 0){
	test = false;
      } else if (sec % (trainSecs + testSecs) - trainSecs == 0){
	test = true;
      }
      t = 0;
      while (t<1000 && !done)				// simulation of 1 sec
	{
	  if (!fileInput){
	    for (i=0;i<N;i++){
	      I[i] = 0.0;	// reset the input
	    }
	    for (k=0;k<N/1000;k++){
	      I[getrandom(N)]=20.0; // random thalamic input
	    }
	  } else { // file input
	    int ii;
	    for (ii=feat;ii<N;ii++){
	      I[ii] = 0.0;
	    }
	    //parse input
	    size_t next = 0;
	    size_t last = 0;
	    for (ii=0;ii<feat;ii++){
	      next = currentLine.find(" ",last);
	      if (ii<inFeat){ //input neurons
		I[ii] = stof(currentLine.substr(last,next-last)) * scale ;
	      } else { //label neurons
		if (test){
		  int q = 0;
		  I[ii] = 0.0;
		  //check for active label
		  //if same as last time, leave it, otherwise switch
		  if (stof(currentLine.substr(last,next-last)) > 0.5 && ii != lastLabel){
		    labelData << "sec= " << sec << ", Label=" << lastLabel - inFeat << ", [ ";
		    for (q=0;q<numLabels;q++){
		      labelData << labelSpikes[q] << " ";
		    }
		    labelData << "]" << endl;
		    for (q=0;q<numLabels;q++){
		      labelSpikes[q] = 0;
		    }
		    lastLabel = ii;
		  }
		} else {
		  I[ii] = (stof(currentLine.substr(last,next-last)) - shift) * labelScale;
		}
	      }
	      last = next + 1;
	      //inputLine.erase(0,pos);
	    }
	    if (framesLeft == 0){
	      //load next line (or stop)
	      currentLine = inputLine;
      	      getline(inputData, inputLine);
	      if (inputData.eof()){
		preDone = true;
	      }
	      //reset framesLeft
	      framesLeft = step -1;
	    } else {
	      framesLeft--;
	      if (preDone && framesLeft == 0){
		done = true;
	      }
	    }
	  }
	  for (i=0;i<N;i++) {
	    if (v[i]>=30)    // did it fire?
	      {
		v[i] = -65.0;	// voltage reset
		u[i]+=d[i];	// recovery variable reset
		LTP[i][t+D]= 0.1;		
		LTD[i]=0.12;
		for (j=0;j<N_pre[i];j++){
		  *sd_pre[i][j]+=LTP[I_pre[i][j]][t+D-D_pre[i][j]-1];
		  // this spike was after pre-synaptic spikes
		}
		firings[N_firings  ][0]=t;
		firings[N_firings++][1]=i;
		if (N_firings == N_firings_max) {
		  cout << "Too many spikes at t=" << t << " (ignoring all)" << endl;
		  N_firings=1;
		}
		if (test && inFeat <= i && i < feat){
		  labelSpikes[i-inFeat]++;
		}
	      }
	  }
	  k=N_firings;
	  while (t-firings[--k][0] <D)
	    {
	      for (j=0; j< delays_length[firings[k][1]][t-firings[k][0]]; j++)
		{
		  i=post[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]]; 
		  I[i]+=s[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]];
		  if (firings[k][1] <Ne) // this spike is before postsynaptic spikes
		    sd[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]]-=LTD[i];
		}
	    }
	  for (i=0;i<N;i++)
	    {
	      v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // for numerical stability
	      v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // time step is 0.5 ms
	      u[i]+=a[i]*(0.2*v[i]-u[i]);
	      LTP[i][t+D+1]=0.95*LTP[i][t+D];
	      LTD[i]*=0.95;
	    }
	  t++;
	}
      cout << "sec=" << sec << ", firing rate=" << float(N_firings)/N << "\n";
      //if (sec == 0){
      fs = fopen("spikes.dat","w");
      for (i=1;i<N_firings;i++){
	if (firings[i][0] >=0){
	  fprintf(fs, "%d  %d\n", firings[i][0], firings[i][1]);
	}
      }
      fclose(fs);
      //}
      for (i=0;i<N;i++){		// prepare for the next sec
	for (j=0;j<D+1;j++){
	  LTP[i][j]=LTP[i][1000+j];
	}
      }
      k=N_firings-1;
      while (1000-firings[k][0]<D){
	k--;
      }
      for (i=1;i<N_firings-k;i++)
	{
	  firings[i][0]=firings[k+i][0]-1000;
	  firings[i][1]=firings[k+i][1];
	}
      N_firings = N_firings-k;
      
      for (i=0;i<Ne;i++){	// modify only exc connections
	for (j=0;j<M;j++)
	  {
	    s[i][j]+=0.01+sd[i][j];
	    sd[i][j]*=0.9;			
	    if (s[i][j]>sm) s[i][j]=sm;
	    if (s[i][j]<0) s[i][j]=0.0;
	  }
      }
      sec++;
      if (sec == maxSecs){
	done = true;
      }
    }
  if (inputData.is_open()){
    inputData.close();
  }
  return 0;
}
