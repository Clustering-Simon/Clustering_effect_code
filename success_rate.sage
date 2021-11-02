#! /usr/bin/env sage

# Accompanying code to the paper "Clustering Effect in Simon and Simeck"
# Computes the success probabilty of linear attacks with various parameters

log2 = lambda x: log(1.0*x)/log(2.0)
Phi = lambda x: (1 +erf(x/sqrt(2 )))/2 
Phi_inv = lambda x: sqrt(2 )*erfinv(2 *x-1 )

distinguishers = {
    'Simeck64 29r (5 approx)': (64 , 2^63 , 52 , [2^-58.47 ]+([2^-60.36]*4 )),
    'Simeck64 29r': (64, 2^62,   26, [2^-58.47]),
    'Simeck64 30r A': (64, 2^63.5, 24 , [2^-60.36 ]),
    'Simeck64 30r B': (64, 2^64,   29 , [2^-60.36 ]),
    'Simeck48 21r': (48, 2^47,   26, [2^-43.56]),
    'Simeck32 13r': (32, 2^31.5,   37, [2^-27.68]),

    'Simon128 41r': (128, 2^126, 10, [2^-123.07]),
    'Simon128 42r': (128, 2^127, 10, [2^-125.07]),
    'Simon96  33r A': (96, 2^95, 10, [2^-92.60]),
    'Simon96  33r B': (96, 2^94, 10, [2^-92.60]),

for d in distinguishers.keys():

    print ("## "+d)
    n, N, a, M = distinguishers[d]

    if n==32:
        # ELP for n==32 is exact
        expC = sum(m for m in M)
        varC = 2*sum(m^2 for m in M)

    else:
        # add 2^-n to ELP
        expC = sum(m+2^-n for m in M)
        varC = 2*sum((m+2^-n)^2 for m in M)
    
    print ("Parameters: n={} N=2^{:.1f} a={} M={} C={:.2f} C'={:.2f}".format(n, log2(N), a, len(M), log2(sum(M)), log2(expC)))

    B = (2^n-N)/(2^n-1)     # Distinct plaintexts
    # B = 1                 # Random plaintext
    
    if len(M) == 1:
        # Single approximation

        sr = B/N+expC
        sw = B/N+2^-n

        p = numerical_approx(2  - 2 *Phi(sqrt(sw/sr)*Phi_inv(1 -2^(-a-1 ))), prec=112 )
        print ("Succes probability (direct)           : {}".format(p))

    else:
        # Multiple approximations
        mr = B*len(M) + N*expC
        sr = 2*B^2*len(M) + 4*B*N*expC + N^2*varC
        Right = lambda x: RealDistribution('gaussian',sqrt(sr)).cum_distribution_function(x-mr)
    
        w = B+N*2^-n
        Wrong = lambda x: RealDistribution('chisquared',len(M)).cum_distribution_function_inv(x)*w
    
        p = 1 - Right(Wrong(1 - 2^-a))
        print ("Succes probability (gaussian/chi_2)   : {}".format(p))
