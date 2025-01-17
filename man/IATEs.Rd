% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/IATEs.R
\name{IATEs}
\alias{IATEs}
\title{This function produces the predicted values of all 11 estimators}
\usage{
IATEs(y_t, d_t, x_t, x_v, K = 2)
}
\arguments{
\item{y_t}{Vector of training outcome values}

\item{d_t}{Vector of training treament indicators}

\item{x_t}{Matrix of training covariates (N x p matrix)}

\item{x_v}{Matrix of validation covariates (J x p, validation sample)}

\item{K}{Integer of the number of splits}
}
\value{
Returns a list with length 11 which contains the IATEs of the validation sample
}
\description{
This function produces the predicted values of all 11 estimators
}
