data {
    int<lower=0> N; // NUMBER OF SEASONS
    int y[N]; // HISTORICAL DATA
    real<lower=0> alpha; // PARAMETER FOR GAMMA DISTRIBUTION
    real<lower=0> beta; // PARAMETER FOR GAMMA DISTRIBUTION
}
parameters {
    real<lower=0> la; // LAMBDA FOR POISSON
}
model {
    target += gamma_lpdf(la | alpha, beta); // PRIOR
    target += poisson_lpmf(y | la); // LIKELIHOOD
}
generated quantities {
    real pred = poisson_rng(la); // PREDICTION FOR CR7

    vector[N] log_lik;
      for (i in 1:N)
        log_lik[i] = poisson_lpmf(y[i] | la);
}