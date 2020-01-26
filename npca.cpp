#include	<Eigen/Eigen>
#include	<algorithm>
#include	"wymlp.hpp"
#include	<iostream>
#include	<fstream>
#include	<unistd.h>
#include	<cfloat>
#include	<vector>
#include	<cmath>
#include	<omp.h>
using	namespace	std;
using	namespace	Eigen;

bool	load_matrix(const	char	*F,	vector<npca_type>	&M,	uint64_t	&R,	uint64_t	&C) {
	ifstream	fi(F);
	if(!fi) {	cerr<<"fail to open "<<F<<'\n';	return	false;	}
	string	buf;	R=C=0;
	while(getline(fi,buf))	if(buf.size()) {
		char	*p=(char*)buf.data(),	*q;
		for(;;) {	
			q=p;	npca_type	x=strtod(p,	&p);
			if(p!=q)	M.push_back(x);	else	break;
		}
		R++;
	}
	fi.close();
	if(M.size()%R) {	cerr<<"unequal column\t"<<F<<'\n';	return	false;	}
	C=M.size()/R;	cerr<<F<<'\t'<<R<<'*'<<C<<'\n';
	return	true;
}

void	document(void){
	cerr<<"usage:	npca input [options]\n";
	cerr<<"\t-c:	covariance=off\n";
	cerr<<"\t-l:	loss function=1(0-3)\n";
	cerr<<"\t-e:	learning rate=0.001\n";
	cerr<<"\t-o:	output=pc\n";
	cerr<<"\t-s:	random number seed=0\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	size_t	t0=time(NULL);
	cerr<<"\n=======================================\n";
	cerr<<"| Neural Principal Component Analysis |\n";
	cerr<<"| author: Yi Wang                     |\n";
	cerr<<"| email:  godspeed_china@yeah.net     |\n";
	cerr<<"| date:   25/Jan/2020                 |\n";
	cerr<<"=======================================\n";
	string	out="pc";
	npca_type	eta=0.001;
	uint64_t	seed=0,	task=1,	cov=0;;
	int	opt;
	while((opt=getopt(ac,	av,	"cl:e:o:s:"))>=0){
		switch(opt){
		case	'c':	cov=1;	break;
		case	'l':	task=atoi(optarg);	break;
		case	'e':	eta=atof(optarg);	break;
		case	'o':	out=optarg;	break;
		case	's':	seed=atoi(optarg);	break;
		default:	document();
		}
	}
	if(ac<optind+1)	document();
	vector<npca_type>	data;	uint64_t	sample,	feature;
	if(!load_matrix(av[optind],	data,	sample,	feature))	return	0;
	cerr<<"output\t"<<out<<'\n';
	cerr<<"use_cov\t"<<cov<<'\n';
	cerr<<"eta\t"<<eta<<'\n';
	cerr<<"network\t"<<feature<<" => 3 => "<<npca_hidden<<"*"<<npca_depth-1<<" => "<<feature<<"\tL"<<task<<"_loss\n";
	for(size_t	j=0;	j<feature;	j++){
		double	sx=0,	sxx=0;
		for(size_t	i=0;	i<sample;	i++){	double	x=data[i*feature+j];	sx+=x;	sxx+=x*x;	}
		sx/=sample;	sxx=sxx/sample-sx*sx;	if(sxx>0)	sxx=1/sqrt(sxx);	else	sxx=0;	if(cov)	sxx=1;
		for(size_t	i=0;	i<sample;	i++)	data[i*feature+j]=(data[i*feature+j]-sx)*sxx;
	}
	vector<npca_type>	w(wymlp<npca_type,npca_hidden,npca_depth>(feature,task,NULL,NULL,NULL,0,0,0));
	uint64_t	rng=wyhash64(0,seed);	for(size_t	i=0;	i<w.size();	i++)	w[i]=wy2gau(wyrand(&rng));
	{
		Map<Matrix<float,-1,-1,RowMajor>	>	m(data.data(),sample,feature);
//		for(size_t  i=0;    i<sample;   i++)	for(size_t  j=0;    j<feature;  j++)	m(i,j)=data[i*feature+j];
		BDCSVD<Matrix<float,-1,-1,RowMajor>	>	svd(m,	ComputeThinU|ComputeThinV);
		string	fn=out;	fn+=".pca";
		ofstream	fo(fn.c_str());
		for(size_t	i=0;	i<sample;	i++)	fo<<svd.matrixU()(i,0)<<'\t'<<svd.matrixU()(i,1)<<'\n';
		fo.close();
		for(size_t	i=0;	i<feature;	i++){	w[i*2]=svd.matrixV()(i,0);	w[i*2+1]=svd.matrixV()(i,1);	}
	}	
	double	drop=exp(-1),	loss0,	loss=FLT_MAX;	size_t	gr=0;	cerr.precision(5);	cerr.setf(ios::fixed);
	for(size_t	it=0;	gr<4;	it++){
		loss0=loss;	loss=0;
		for(size_t	i=0;	i<0x100000;	i++){
			uint64_t	j=wyrand(&rng)%sample;
			loss+=wymlp<npca_type,npca_hidden,npca_depth>(feature,task,w.data(),data.data()+j*feature,data.data()+j*feature,eta,wyrand(&rng),drop);
		}
		if(loss>loss0)	gr++;
		cerr<<loss/0x100000/feature/drop<<'\t';
	}
	string	fn=out;	fn+=".npca";
	ofstream	fo(fn.c_str());
	for(size_t	i=0;	i<sample;	i++){
		npca_type	y[2]={};
		wymlp<npca_type,npca_hidden,npca_depth>(feature,task,w.data(),data.data()+i*feature,y,-1,0,drop);
		fo<<y[0]<<'\t'<<y[1]<<'\n';
	}
	fo.close();
	fn=out;	fn+=".npca_loading";
	for(size_t	i=0;	i<feature;	i++)	fo<<w[i*2]<<'\t'<<w[i*2+1]<<'\n';
	fo.close();
	cerr<<"\ntime:\t"<<time(NULL)-t0<<" sec\n";
	return	0;
}
