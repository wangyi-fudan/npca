#include	"wyhash.h"
#include	<math.h>
template<class	type,	unsigned	hidden,	unsigned	depth,	unsigned	task>
type	wymlp(unsigned	input,	type	*weight,	type	*x,	type	*y,	type	eta,	uint64_t	seed,	double	dropout) {
	if(weight==NULL)	return	(input+1)*hidden+(depth-1)*hidden*hidden+input*hidden;
	#define	woff(i,l)	(l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden)
	#define	wymlp_act(x)	(x/(1+(((int)(x>0)<<1)-1)*x))
	#define	wymlp_gra(x)	((1-(((int)(x>0)<<1)-1)*x)*(1-(((int)(x>0)<<1)-1)*x))
	type	a[2*depth*hidden+input]= {},	*d=a+depth*hidden,	*o=a+2*depth*hidden,	wh=1/sqrtf(hidden),	wi=(1-(eta<0)*dropout)/sqrtf(input+1);	uint64_t	drop=dropout*~0ull;
	for(unsigned  i=0;  i<=input; i++)	{
		type	*w=weight+woff(i,0),	s=(i==input?1:x[i])*(eta<0||wyhash64(i,seed)>=drop);
		for(unsigned	j=0;	j<2;	j++)	a[j]+=s*w[j];
	}
	a[0]*=wi;	a[1]*=wi;	a[2]=1;	
	if(eta<0){	y[0]=a[0]*wi;	y[1]=a[1]*wi;	return	0;	}
	for(unsigned	l=1;	l<=depth;	l++) {
		type	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
		for(unsigned	i=0;	i<(l==depth?input:hidden);	i++) {
			type	*w=weight+woff(i,l),	s=0;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			q[i]=(l==depth?s*wh:(i?wymlp_act(s*wh):1));
		}
	}
	type	loss=0;
	switch(task) {
	case	0:	{	
		for(unsigned	i=0;	i<input;	i++)	
			if(wyhash64(i,seed)<drop){	o[i]=1/(1+expf(-o[i]))-y[i];	loss+=o[i]*o[i];	o[i]*=eta;	}	
			else	o[i]=0;	
	}	break;
	case	1:	{	
		for(unsigned	i=0;	i<input;	i++)	
		if(wyhash64(i,seed)<drop){	o[i]-=y[i];	loss+=o[i]>0?o[i]:-o[i];	o[i]=o[i]>0?eta:-eta;	}	
		else	o[i]=0;	
	}	break;
	case	2:	{	
		for(unsigned	i=0;	i<input;	i++)	
			if(wyhash64(i,seed)<drop){	o[i]-=y[i];	loss+=o[i]*o[i];	o[i]*=eta;	}	
			else	o[i]=0;	
	}	break;
	case	3:	{	
		for(unsigned	i=0;	i<input;	i++)	
			if(wyhash64(i,seed)<drop){	o[i]-=y[i];	loss+=o[i]>0?o[i]*o[i]*o[i]:-o[i]*o[i]*o[i];	o[i]=o[i]>0?o[i]*o[i]*eta:-o[i]*o[i]*eta;	}	
			else	o[i]=0;	
	}	break;
	}
	for(unsigned	l=depth;	l;	l--) {
		type	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden),	*g=d+(l-1)*hidden,	*h=(l==depth?o:d+l*hidden);
		for(unsigned	i=0;	i<(l==depth?input:hidden);	i++) {
			type	*w=weight+woff(i,l),	s=(l==depth?q[i]:h[i]*wymlp_gra(q[i]))*wh;
			for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
		}
	}
	d[0]*=wi;	d[1]*=wi;
	for(unsigned  i=0;  i<=input; i++)	{
		type	*w=weight+woff(i,0),	s=(i==input?1:x[i])*(eta<0||wyhash64(i,seed)>=drop);
		for(unsigned	j=0;	j<2;	j++)	w[j]-=s*d[j];
	}
	return	loss;
}
