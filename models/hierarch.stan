data {
    int<lower=0> N; // NUMBER OF SEASONS (3)
    int<lower=0> J; // NUMBER OF EPL FORWARDS (3)
    vector[J] y[N]; // GOAL DIFFERENCES
}
parameters {
    real tau; // HYPERPARAMETER
    real<lower=0> sigma0; // HYPERPARAMETER, CANNOT BE NEGATIVE
    vector[J] mu; // MEAN FOR EACH PLAYER
    real<lower=0> sigma; // COMMON VARIANCE, CANNOT BE NEGATIVE 
}
model {
    tau ~ normal(0, 10); // HYPERPRIOR
    sigma0 ~ inv_chi_square(0.1); // HYPERPRIOR
    sigma ~ inv_chi_square(0.1); // COMMON VARIANCE FOR ALL GROUPS
    for (j in 1:J) {
        mu[j] ~ normal(tau, sigma0);
        y[, j] ~ normal(mu[j], sigma); // LIKELIHOOD
    }
}
generated quantities {
    real mu4 = normal_rng(tau, sigma0);
    real pred4 = normal_rng(mu4, sigma); // PREDICTION
    vector[N*J] log_lik;
    for (j in 1:J)
      for (i in 1:N)
        log_lik[(i - 1)*J + j] = normal_lpdf(y[i,j] | mu[j], sigma);
}