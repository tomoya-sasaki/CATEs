% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CATEs_utils.R
\name{cf_dml2}
\alias{cf_dml2}
\title{This function implements the cross-fitting}
\usage{
cf_dml2(est, y, d, x, np, xnew, index, args_tau = list())
}
\arguments{
\item{est}{Estimation strategy}

\item{y}{Vector of outcome values}

\item{d}{Vector of treatment indicators}

\item{x}{Matrix of covariates (N x p matrix)}

\item{np}{Matrix of nuisance parameters}

\item{xnew}{Matrix of covariates (J x P matrix)}

\item{index}{List indicating indices for cross-fitting (e.g. obtained by \code{createFolds} of \code{\link[pkg]{caret}} pkg)}

\item{args_tau}{List of arguments passed to estimate IATEs#'}
}
\value{
Returns J x 4 matrix containing the nuisance parameters
}
\description{
This function implements the cross-fitting
}
