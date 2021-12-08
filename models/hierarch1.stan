data {
    int<lower=0> N; // NUMBER OF SEASONS (12)
    int<lower=0> J; // NUMBER OF ManU FORWARDS (3)
    vector[J] y1[N]; // HISTORICAL DATA FOR THE ManU FORWARDS' GOAL SCORING
    vector[N] y2; // DATA FOR CR7
}

parameters {
    real<lower=0> mu; // HYPERPARAMETER, CANNOT BE NEGATIVE
    real<lower=0> tau; // HYPERPARAMETER, CANNOT BE NEGATIVE
    vector<lower=0>[J] theta; // PRIOR FOR EACH GROUP, CANNOT BE NEGATIVE
    real<lower=0> sigma; // PRIOR, CANNOT BE NEGATIVE 
}

model {
    mu ~ normal(0, 100); // HYPERPRIOR
    tau ~ inv_chi_square(0.1); // HYPERPRIOR
    sigma ~ inv_chi_square(0.1); // COMMON STD FOR ALL GROUPS
    for (j in 1:J) {
        theta[j] ~ normal(mu, tau);
        y1[, j] ~ poisson (theta[j]); // LIKELIHOOD
    }
}

generated quantities {
    real theta_4 = normal_rng(mu, tau);
    real pred4 = normal_rng(theta_4, sigma); // PRIOR FOR CR7

    vector[N*J] log_lik;
    for (j in 1:J)
      for (i in 1:N)
        log_lik[(i - 1)*J + j] = normal_lpdf(y1[i,j] | theta[j], sigma);
}